#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

import numpy as np
import optuna
import scipy.stats as st

from quantum_estimator import ProbEstimate
from quantum_iqae import estimate_var_quantum
TailMode = Literal["pnl_leq", "loss_geq"]


@dataclass(frozen=True)
class TailQuery:
    alpha_target: float
    tail_mode: TailMode
    threshold_value: Optional[float] = None
    index: Optional[int] = None
    grid_points: Optional[np.ndarray] = None


def get_log_normal_probabilities(mu, sigma, num_points):
    """Existing log-normal implementation."""
    mean = np.exp(mu + sigma ** 2 / 2)
    var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
    std = np.sqrt(var)
    low = max(0.0, mean - 3 * std)
    high = mean + 3 * std
    x = np.linspace(low, high, num_points)
    pdf = st.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    p = pdf / np.sum(pdf)
    return x, p


def get_normal_probabilities(mu, sigma, num_points):
    """Discretized Gaussian."""
    low = mu - 4 * sigma
    high = mu + 4 * sigma
    x = np.linspace(low, high, num_points)
    pdf = st.norm.pdf(x, loc=mu, scale=sigma)
    p = pdf / np.sum(pdf)
    return x, p


def get_exponential_probabilities(lam, num_points):
    """Discretized exponential with rate lambda."""
    mean = 1.0 / lam
    high = mean + 5 * mean  # covers ~99.3%
    x = np.linspace(0.0, high, num_points)
    pdf = st.expon.pdf(x, scale=1.0 / lam)
    p = pdf / np.sum(pdf)
    return x, p


def get_pareto_probabilities(alpha_shape, x_m, num_points):
    """Discretized Pareto (heavy tail). alpha_shape=shape, x_m=scale."""
    # Pareto has infinite mean for alpha<=1, cap at 99th percentile
    high = st.pareto.ppf(0.99, b=alpha_shape, scale=x_m)
    x = np.linspace(x_m, high, num_points)
    pdf = st.pareto.pdf(x, b=alpha_shape, scale=x_m)
    p = pdf / np.sum(pdf)
    return x, p


def get_mixture_probabilities(num_points):
    """Gaussian mixture: 80% N(1, 0.2) + 20% N(3, 0.5) — bimodal."""
    x = np.linspace(-1, 5, num_points)
    pdf = 0.8 * st.norm.pdf(x, loc=1, scale=0.2) + 0.2 * st.norm.pdf(x, loc=3, scale=0.5)
    p = pdf / np.sum(pdf)
    return x, p


def get_beta_probabilities(a, b, num_points):
    """Discretized Beta(a, b) on [0, 1]."""
    x = np.linspace(1e-6, 1 - 1e-6, num_points)
    pdf = st.beta.pdf(x, a, b)
    p = pdf / np.sum(pdf)
    return x, p

def get_student_t_probabilities(df, loc, scale, num_points):
    """Discretized Student's t with df degrees of freedom."""
    # Use PPF to get reasonable bounds (heavy tails need wider range)
    low = st.t.ppf(0.001, df, loc=loc, scale=scale)
    high = st.t.ppf(0.999, df, loc=loc, scale=scale)
    x = np.linspace(low, high, num_points)
    pdf = st.t.pdf(x, df, loc=loc, scale=scale)
    p = pdf / np.sum(pdf)
    return x, p


def get_skew_normal_probabilities(alpha_skew, loc, scale, num_points):
    """Discretized skew-normal. alpha_skew>0 = right skew, <0 = left skew."""
    low = st.skewnorm.ppf(0.001, alpha_skew, loc=loc, scale=scale)
    high = st.skewnorm.ppf(0.999, alpha_skew, loc=loc, scale=scale)
    x = np.linspace(low, high, num_points)
    pdf = st.skewnorm.pdf(x, alpha_skew, loc=loc, scale=scale)
    p = pdf / np.sum(pdf)
    return x, p


def get_distribution(dist_name, num_points, **kwargs):
    """
    Factory function for distributions.
    Returns (grid_points, probabilities).
    """
    if dist_name == "lognormal":
        return get_log_normal_probabilities(
            kwargs.get("mu", 0.7),
            kwargs.get("sigma", 0.13),
            num_points
        )
    elif dist_name == "normal":
        return get_normal_probabilities(
            kwargs.get("mu", 0.0),
            kwargs.get("sigma", 1.0),
            num_points
        )
    elif dist_name == "exponential":
        return get_exponential_probabilities(
            kwargs.get("lam", 1.0),
            num_points
        )
    elif dist_name == "pareto":
        return get_pareto_probabilities(
            kwargs.get("shape", 2.5),
            kwargs.get("scale", 1.0),
            num_points
        )
    elif dist_name == "mixture":
        return get_mixture_probabilities(num_points)
    elif dist_name == "beta":
        return get_beta_probabilities(
            kwargs.get("a", 2.0),
            kwargs.get("b", 5.0),
            num_points
        )
    elif dist_name == "student_t":
        return get_student_t_probabilities(
            kwargs.get("df", 3.0),
            kwargs.get("loc", 0.0),
            kwargs.get("scale", 1.0),
            num_points
        )
    elif dist_name == "skew_normal":
        return get_skew_normal_probabilities(
            kwargs.get("skew", 4.0),
            kwargs.get("loc", 0.0),
            kwargs.get("scale", 1.0),
            num_points
        )
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


def wilson_ci(k: int, n: int, confidence: float):
    z = st.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    ph = k / n
    den = 1.0 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    rad = (z / den) * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return float(max(0.0, cen - rad)), float(min(1.0, cen + rad))


def solve_var_bisect(
        estimator,
        *,
        alpha_target: float,
        tail_mode: TailMode,
        grid_points: np.ndarray,
        lo_index: int,
        hi_index: int,
        prob_tol: float,
        value_tol: float,
        max_steps: int,
        est_params: Dict[str, Any],
):
    total_cost = 0
    lo, hi = lo_index, hi_index
    trace = []
    for _ in range(max_steps):
        if lo >= hi or (grid_points[hi] - grid_points[lo]) <= value_tol:
            break
        mid = (lo + hi) // 2
        q = TailQuery(alpha_target=alpha_target, tail_mode=tail_mode, index=mid, grid_points=grid_points)
        est = estimator(q, **est_params)
        total_cost += est.cost
        trace.append((mid, est.p_hat, est.ci_low, est.ci_high, est.cost))

        # CI-driven move
        if alpha_target > est.ci_high + prob_tol:
            lo = mid + 1
        elif alpha_target < est.ci_low - prob_tol:
            hi = mid
        else:
            # ambiguous: fallback on point estimate
            if est.p_hat < alpha_target:
                lo = mid + 1
            else:
                hi = mid

    var_index = lo
    return float(grid_points[var_index]), int(var_index), int(total_cost), trace


# Classical estimator (advanced discrete Monte Carlo)


def _validate_pmf(probs: np.ndarray) -> np.ndarray:
    p = np.array(probs, dtype=float).copy()
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probs must be a non-empty 1D array")
    if np.any(p < 0):
        raise ValueError("probs must be non-negative")
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("probs must sum to > 0")
    p /= s
    return p


def _cdf_from_pmf(p: np.ndarray) -> np.ndarray:
    c = np.cumsum(p)
    c[-1] = 1.0
    return c


def _inverse_cdf_sample(u: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    return np.searchsorted(cdf, u, side="left").astype(int)


def _halton_range(start_index_1based: int, n: int, base: int, scramble_seed: Optional[int]) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=float)
    if start_index_1based <= 0:
        raise ValueError("start_index_1based must be >= 1")

    perm = None
    if scramble_seed is not None:
        rng = np.random.default_rng(scramble_seed)
        perm = np.arange(base, dtype=int)
        rng.shuffle(perm)

    out = np.zeros(n, dtype=float)
    for j in range(n):
        x = start_index_1based + j
        f = 1.0
        r = 0.0
        while x > 0:
            f /= base
            digit = x % base
            if perm is not None:
                digit = int(perm[digit])
            r += f * digit
            x //= base
        out[j] = r
    return out


def _get_index_threshold(query: TailQuery) -> int:
    if getattr(query, "index", None) is None:
        raise ValueError("Estimator expects TailQuery.index for discrete CDF queries.")
    return int(query.index)


def _wilson_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    return wilson_ci(k, n, confidence)


@dataclass(frozen=True)
class _CacheKey:
    reuse_id: str
    seed: Optional[int]
    method: str
    tilt_tau: float
    qmc: bool
    scramble_seed: Optional[int]
    strata: int
    pilot: int


class ClassicalDiscreteMC:
    """
    Advanced discrete Monte Carlo estimator with:
      - IS (exponential tilting), stratification, QMC, control variates
      - Option B caching: cached sample set grows to satisfy larger budgets safely
      - cost returned is NEW samples generated in the call

    Tail event is P(bin < query.index).
    """

    def __init__(self, probs: np.ndarray):
        self._p = _validate_pmf(np.array(probs, dtype=float))
        self._support = np.arange(self._p.size, dtype=int)
        self._cdf_p = _cdf_from_pmf(self._p)
        self._EX_index = float(np.sum(self._support * self._p))
        self._cache: Dict[_CacheKey, Dict[str, Any]] = {}

        try:
            import torch
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                self.p_t = torch.tensor(self._p, device=self.device, dtype=torch.float32)
                self.support_t = torch.arange(self._p.size, device=self.device, dtype=torch.long)
            else:
                self.p_t = None
                self.support_t = None
        except Exception:
            self.torch = None
            self.device = "cpu"
            self.p_t = None
            self.support_t = None

    def _proposal_from_tilt(self, tau: float) -> np.ndarray:
        tau = float(tau)
        if tau < 0:
            raise ValueError("tilt_tau must be >= 0")
        logw = -tau * self._support.astype(float)
        q_unnorm = self._p * np.exp(logw - np.max(logw))
        return _validate_pmf(q_unnorm)

    def _make_strata(self, num_strata: int) -> list[tuple[int, int, float]]:
        k = int(num_strata)
        if k <= 1:
            return [(0, self._p.size, 1.0)]

        edges = [0]
        targets = [(j / k) for j in range(1, k)]
        for t in targets:
            idx = int(np.searchsorted(self._cdf_p, t, side="left"))
            edges.append(idx)
        edges.append(self._p.size)

        strata = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            mass = float(np.sum(self._p[a:b]))
            strata.append((a, b, mass))
        return strata if strata else [(0, self._p.size, 1.0)]

    def _draw_u(
            self,
            *,
            rng: np.random.Generator,
            m: int,
            base: int,
            qmc: bool,
            scramble_seed: Optional[int],
            qmc_counters: Dict[int, int],
    ) -> np.ndarray:
        if m <= 0:
            return np.empty((0,), dtype=float)
        if not qmc:
            return rng.random(m)

        used = int(qmc_counters.get(base, 0))
        start = used + 1
        u = _halton_range(start, m, base=base, scramble_seed=scramble_seed)
        qmc_counters[base] = used + m
        return u

    def _draw_indices_batch(
            self,
            *,
            rng: np.random.Generator,
            n: int,
            method: str,
            tilt_tau: float,
            qmc: bool,
            scramble_seed: Optional[int],
            strata: int,
            pilot: int,
            qmc_counters: Dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        n = int(n)
        if n <= 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=float), {"n": 0}

        method = str(method)
        qmc = bool(qmc)
        tilt_tau = float(tilt_tau)
        strata = int(strata)
        pilot = int(pilot)

        if method in ("plain", "stratified", "qmc", "stratified_qmc"):
            q = self._p
            cdf_q = self._cdf_p
            use_is = False
        elif method in ("is", "is_stratified", "is_qmc", "is_stratified_qmc"):
            q = self._proposal_from_tilt(tilt_tau)
            cdf_q = _cdf_from_pmf(q)
            use_is = True
        else:
            raise ValueError(f"Unknown method: {method}")

        meta: Dict[str, Any] = {
            "method": method,
            "qmc": qmc,
            "tilt_tau": tilt_tau,
            "use_is": use_is,
            "strata": strata,
            "pilot": pilot,
            "scramble_seed": scramble_seed,
        }

        if strata <= 1:
            u = self._draw_u(
                rng=rng,
                m=n,
                base=2,
                qmc=qmc,
                scramble_seed=scramble_seed,
                qmc_counters=qmc_counters,
            )
            idx = _inverse_cdf_sample(u, cdf_q)
            w = (self._p[idx] / q[idx]) if use_is else np.ones(n, dtype=float)
            return idx, w, meta

        S = self._make_strata(strata)
        masses = np.array([m for _, _, m in S], dtype=float)
        masses = masses / float(np.sum(masses))

        alloc = np.maximum(1, (n * masses).astype(int))
        while int(np.sum(alloc)) > n:
            alloc[np.argmax(alloc)] -= 1
        while int(np.sum(alloc)) < n:
            alloc[np.argmin(alloc)] += 1
        meta["mass_alloc"] = alloc.tolist()

        idx_parts = []
        w_parts = []
        for s, (lo, hi, _) in enumerate(S):
            m = int(alloc[s])
            if m <= 0:
                continue

            q_slice = q[lo:hi]
            q_slice = _validate_pmf(q_slice)
            cdf_slice = _cdf_from_pmf(q_slice)

            base = 2 + (s % 5)
            u = self._draw_u(
                rng=rng,
                m=m,
                base=base,
                qmc=qmc,
                scramble_seed=scramble_seed,
                qmc_counters=qmc_counters,
            )
            idx_local = _inverse_cdf_sample(u, cdf_slice) + lo
            idx_parts.append(idx_local)
            w_parts.append((self._p[idx_local] / q[idx_local]) if use_is else np.ones(m, dtype=float))

        idx = np.concatenate(idx_parts)[:n]
        w = np.concatenate(w_parts)[:n]
        return idx, w, meta

    def _estimate_from_samples(
            self,
            *,
            idx: np.ndarray,
            w: np.ndarray,
            index_thr: int,
            confidence: float,
            use_cv: bool,
    ) -> tuple[float, float, float, float, Dict[str, Any]]:
        n = int(idx.size)
        if n <= 0:
            return 0.0, 0.0, 1.0, 0.0, {"cv_used": False}

        y = (idx < int(index_thr)).astype(float)
        w = np.asarray(w, dtype=float)

        sum_w = float(np.sum(w))
        if sum_w <= 0:
            raise ValueError("sum of weights must be > 0")

        if use_cv:
            x = idx.astype(float)
            EX = self._EX_index

            a = w / sum_w
            mean_y = float(np.sum(a * y))
            mean_x = float(np.sum(a * x))

            cov_xy = float(np.sum(a * (x - mean_x) * (y - mean_y)))
            var_x = float(np.sum(a * (x - mean_x) ** 2)) + 1e-18
            beta = cov_xy / var_x

            y = y - beta * (x - EX)

            cv_meta = {
                "cv_used": True,
                "cv_beta": float(beta),
                "cv_EX_index": float(EX),
                "cv_mean_index_weighted": float(mean_x),
            }
        else:
            cv_meta = {"cv_used": False}

        phat = float(np.sum(w * y) / sum_w)

        ess = (sum_w * sum_w) / float(np.sum(w * w) + 1e-18)
        ess_int = max(1, int(ess))
        successes_eff = phat * ess
        k_eff = int(round(successes_eff))

        ci_low, ci_high = _wilson_ci(k_eff, ess_int, confidence)
        return phat, float(ci_low), float(ci_high), float(successes_eff), cv_meta

    def estimate_tail_prob(
            self,
            query: TailQuery,
            *,
            budget: int,
            confidence: float = 0.99,
            seed: Optional[int] = None,
            **kwargs: Any,
    ) -> ProbEstimate:
        index_thr = _get_index_threshold(query)

        n_req = int(budget)
        if n_req <= 0:
            raise ValueError("budget must be positive")

        method = str(kwargs.get("method", "plain"))
        tilt_tau = float(kwargs.get("tilt_tau", 0.0))
        qmc = bool(kwargs.get("qmc", "qmc" in method))
        scramble_seed = kwargs.get("scramble_seed", None)
        strata = int(kwargs.get("strata", 1))
        pilot = int(kwargs.get("pilot", 0))
        use_cv = bool(kwargs.get("use_control_variate", False))

        batch_size = int(kwargs.get("batch_size", 8192))
        batch_size = max(256, batch_size)

        target_prob = kwargs.get("target_prob", None)
        if target_prob is not None:
            target_prob = float(target_prob)
        ci_width_tol = kwargs.get("ci_width_tol", None)
        if ci_width_tol is not None:
            ci_width_tol = float(ci_width_tol)

        reuse_id = kwargs.get("reuse_id", None)
        cache_key = None

        use_fast_path = (
                method == "plain"
                and not qmc
                and float(tilt_tau) == 0.0
                and int(strata) <= 1
                and not use_cv
                and reuse_id is None
                and target_prob is None
                and ci_width_tol is None
        )
        if use_fast_path:
            if self.torch is not None and self.device == "cuda":
                g = self.torch.Generator(device=self.device)
                if seed is not None:
                    g.manual_seed(int(seed))
                samples = self.torch.multinomial(self.p_t, n_req, replacement=True, generator=g)
                k = int((samples < int(index_thr)).sum().item())
            else:
                rng = np.random.default_rng(seed)
                samples = rng.choice(self._support, size=n_req, p=self._p)
                k = int(np.sum(samples < int(index_thr)))

            p_hat = k / n_req
            ci_low, ci_high = _wilson_ci(k, n_req, confidence)
            return ProbEstimate(
                p_hat=float(p_hat),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                cost=int(n_req),
                meta={
                    "device": self.device,
                    "k": int(k),
                    "n": int(n_req),
                    "method": "plain",
                    "fast_path": True,
                    "seed": seed,
                    "confidence": float(confidence),
                },
            )

        use_cuda_is_path = (
                method == "is"
                and self.torch is not None
                and self.device == "cuda"
                and not qmc
                and float(tilt_tau) > 0.0
                and int(strata) <= 1
                and not use_cv
                and reuse_id is None
                and target_prob is None
                and ci_width_tol is None
        )
        if use_cuda_is_path:
            g = self.torch.Generator(device=self.device)
            if seed is not None:
                g.manual_seed(int(seed))

            tau_t = self.torch.tensor(float(tilt_tau), device=self.device, dtype=self.p_t.dtype)
            logw = -tau_t * self.support_t.to(dtype=self.p_t.dtype)
            q_unnorm = self.p_t * self.torch.exp(logw - self.torch.max(logw))
            q_t = q_unnorm / self.torch.sum(q_unnorm)

            samples = self.torch.multinomial(q_t, n_req, replacement=True, generator=g)
            y = (samples < int(index_thr)).to(dtype=self.p_t.dtype)
            w = self.p_t[samples] / q_t[samples]

            sum_w = self.torch.sum(w)
            phat = self.torch.sum(w * y) / sum_w
            ess = (sum_w * sum_w) / self.torch.sum(w * w)

            ess_val = float(ess.item())
            ess_int = max(1, int(ess_val))
            successes_eff = float((phat * ess).item())
            k_eff = int(round(successes_eff))

            ci_low, ci_high = _wilson_ci(k_eff, ess_int, confidence)
            return ProbEstimate(
                p_hat=float(phat.item()),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                cost=int(n_req),
                meta={
                    "device": self.device,
                    "n": int(n_req),
                    "method": "is",
                    "tilt_tau": float(tilt_tau),
                    "use_is": True,
                    "fast_path": True,
                    "seed": seed,
                    "confidence": float(confidence),
                    "successes_eff": float(successes_eff),
                    "ess": float(ess_val),
                },
            )

        if reuse_id is not None:
            cache_key = _CacheKey(
                reuse_id=str(reuse_id),
                seed=seed,
                method=method,
                tilt_tau=tilt_tau,
                qmc=qmc,
                scramble_seed=scramble_seed,
                strata=strata,
                pilot=pilot,
            )

        if cache_key is not None and cache_key in self._cache:
            cached = self._cache[cache_key]
            idx = cached["idx"]
            w = cached["w"]
            rng_state = cached["rng_state"]
            qmc_counters = cached["qmc_counters"]
            base_meta = cached["meta"]
        else:
            idx = np.empty((0,), dtype=int)
            w = np.empty((0,), dtype=float)
            rng_state = None
            qmc_counters: Dict[int, int] = {}
            base_meta: Dict[str, Any] = {}

        new_samples_cost = 0
        if idx.size < n_req:
            rng = np.random.default_rng(seed)
            if rng_state is not None:
                rng.bit_generator.state = rng_state

            decided = False
            while idx.size < n_req and not decided:
                m = min(batch_size, n_req - int(idx.size))

                idx_b, w_b, meta_b = self._draw_indices_batch(
                    rng=rng,
                    n=m,
                    method=method,
                    tilt_tau=tilt_tau,
                    qmc=qmc,
                    scramble_seed=scramble_seed,
                    strata=strata,
                    pilot=pilot,
                    qmc_counters=qmc_counters,
                )
                idx = np.concatenate([idx, idx_b])
                w = np.concatenate([w, w_b])
                new_samples_cost += int(m)
                base_meta = meta_b

                if target_prob is not None or ci_width_tol is not None:
                    phat_tmp, ci_low_tmp, ci_high_tmp, _, _ = self._estimate_from_samples(
                        idx=idx,
                        w=w,
                        index_thr=index_thr,
                        confidence=confidence,
                        use_cv=use_cv,
                    )
                    if target_prob is not None:
                        if ci_high_tmp < target_prob or ci_low_tmp > target_prob:
                            decided = True
                    if ci_width_tol is not None:
                        if (ci_high_tmp - ci_low_tmp) <= ci_width_tol:
                            decided = True

            rng_state = rng.bit_generator.state

        if cache_key is not None:
            self._cache[cache_key] = {
                "idx": idx,
                "w": w,
                "rng_state": rng_state,
                "qmc_counters": dict(qmc_counters),
                "meta": dict(base_meta),
            }

        idx_use = idx[:n_req]
        w_use = w[:n_req]

        phat, ci_low, ci_high, successes_eff, cv_meta = self._estimate_from_samples(
            idx=idx_use,
            w=w_use,
            index_thr=index_thr,
            confidence=confidence,
            use_cv=use_cv,
        )

        meta = {
            "method": str(base_meta.get("method", method)),
            "qmc": bool(base_meta.get("qmc", qmc)),
            "tilt_tau": float(base_meta.get("tilt_tau", tilt_tau)),
            "use_is": bool(base_meta.get("use_is", "is" in method)),
            "strata": int(base_meta.get("strata", strata)),
            "pilot": int(base_meta.get("pilot", pilot)),
            "scramble_seed": base_meta.get("scramble_seed", scramble_seed),
            "seed": seed,
            "n": int(n_req),
            "successes_eff": float(successes_eff),
            "confidence": float(confidence),
            "reuse_id": str(reuse_id) if reuse_id is not None else None,
            "cache_size": int(idx.size),
            "new_samples_this_call": int(new_samples_cost),
        }
        meta.update(cv_meta)

        return ProbEstimate(
            p_hat=float(phat),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            cost=int(new_samples_cost),
            meta=meta,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-in", action="store_true")
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--quantum", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-qubits", type=int, default=7)
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--sigma", type=float, default=0.13)
    ap.add_argument("--alpha", type=float, default=0.07)  # target CDF prob
    ap.add_argument("--prob-tol", type=float, default=None)  # default: alpha/10
    ap.add_argument("--value-tol", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.0)  # VaR abs error tolerance; 0 means pure cost-min
    ap.add_argument("--out", type=str, default="best.json")
    ap.add_argument("--dist", type=str, default="lognormal",
                    choices=["lognormal", "normal", "exponential", "pareto", "mixture", "beta"],
                    help="Distribution type")
    ap.add_argument("--lam", type=float, default=1.0, help="Exponential rate")
    ap.add_argument("--shape", type=float, default=2.5, help="Pareto shape")
    ap.add_argument("--scale", type=float, default=1.0, help="Pareto scale")
    ap.add_argument("--beta-a", type=float, default=2.0)
    ap.add_argument("--beta-b", type=float, default=5.0)
    ap.add_argument("--df", type=float, default=3.0, help="Student-t degrees of freedom")
    ap.add_argument("--skew", type=float, default=4.0, help="Skew-normal skewness parameter")
    ap.add_argument("--loc", type=float, default=0.0, help="Location parameter")
    args = ap.parse_args()

    if args.log_in:
        import classiq
        classiq.authenticate()

    if args.classical == args.quantum:
        raise SystemExit("Pick exactly one: --classical or --quantum")

    grid_points, probs = get_distribution(
        args.dist,
        2 ** args.num_qubits,
        mu=args.mu,
        sigma=args.sigma,
        lam=args.lam,
        shape=args.shape,
        scale=args.scale,
        a=args.beta_a,
        b=args.beta_b,
    )
    cdf = np.cumsum(probs)
    ref_idx = int(np.searchsorted(cdf, args.alpha, side="left"))
    ref_var = float(grid_points[ref_idx])

    prob_tol = args.prob_tol if args.prob_tol is not None else args.alpha / 10.0

    # Estimator selection
    if args.classical:
        # Define test suite of distributions with varying tail behavior
        TEST_DISTRIBUTIONS = [
            {"dist": "lognormal", "mu": 0.7, "sigma": 0.13},
            {"dist": "normal", "mu": 0.0, "sigma": 1.0},
            {"dist": "exponential", "lam": 1.0},
            {"dist": "pareto", "shape": 2.5, "scale": 1.0},
            {"dist": "student_t", "df": 3.0, "loc": 0.0, "scale": 1.0},
            {"dist": "skew_normal", "skew": 4.0, "loc": 0.0, "scale": 1.0},
            {"dist": "beta", "a": 2.0, "b": 5.0},
            {"dist": "mixture"},
        ]

        # Precompute all distributions
        dist_data = []
        for d in TEST_DISTRIBUTIONS:
            gp, pr = get_distribution(
                d["dist"],
                2 ** args.num_qubits,
                mu=d.get("mu", 0.0),
                sigma=d.get("sigma", 1.0),
                lam=d.get("lam", 1.0),
                shape=d.get("shape", 2.5),
                scale=d.get("scale", 1.0),
                a=d.get("a", 2.0),
                b=d.get("b", 5.0),
                df=d.get("df", 3.0),
                skew=d.get("skew", 4.0),
                loc=d.get("loc", 0.0),
            )
            cdf = np.cumsum(pr)
            ref_idx = int(np.searchsorted(cdf, args.alpha, side="left"))
            ref_var = float(gp[ref_idx])
            dist_data.append({
                "name": d["dist"],
                "grid": gp,
                "probs": pr,
                "ref_idx": ref_idx,
                "ref_var": ref_var,
                "estimator": ClassicalDiscreteMC(pr),
            })

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=20)

        def objective(trial: optuna.Trial):
            budget = trial.suggest_int("budget", 500, 50_000, log=True)
            conf = trial.suggest_float("confidence", 0.90, 0.999)

            method = trial.suggest_categorical("method", [
                "plain", "is", "stratified", "qmc",
                "is_stratified", "is_qmc", "is_stratified_qmc"
            ])

            tilt_tau = trial.suggest_float("tilt_tau", 0.01, 2.0) if "is" in method else 0.0
            strata = trial.suggest_int("strata", 4, 64) if "stratified" in method else 1
            use_cv = trial.suggest_categorical("use_control_variate", [True, False])

            NUM_SEEDS = 5  # Average over multiple seeds for stability

            total_cost = 0.0
            total_err = 0.0
            max_err = 0.0

            for i, dd in enumerate(dist_data):
                dist_cost = 0.0
                dist_err = 0.0

                for seed_offset in range(NUM_SEEDS):
                    def estimator(q: TailQuery, **kw):
                        return dd["estimator"].estimate_tail_prob(q, **kw)

                    var, vidx, cost, _ = solve_var_bisect(
                        estimator,
                        alpha_target=args.alpha,
                        tail_mode="pnl_leq",
                        grid_points=dd["grid"],
                        lo_index=0,
                        hi_index=len(dd["grid"]) - 1,
                        prob_tol=prob_tol,
                        value_tol=args.value_tol,
                        max_steps=args.max_steps,
                        est_params={
                            "budget": budget,
                            "confidence": conf,
                            "seed": args.seed + trial.number * 1000 + i * 100 + seed_offset,
                            "method": method,
                            "tilt_tau": tilt_tau,
                            "strata": strata,
                            "use_control_variate": use_cv,
                        },
                    )

                    err = abs(var - dd["ref_var"]) / (abs(dd["ref_var"]) + 1e-9)
                    dist_cost += cost
                    dist_err += err

                # Average over seeds
                dist_cost /= NUM_SEEDS
                dist_err /= NUM_SEEDS

                total_cost += dist_cost
                total_err += dist_err
                max_err = max(max_err, dist_err)

                trial.report(total_cost / (i + 1), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            avg_cost = total_cost / len(dist_data)
            avg_err = total_err / len(dist_data)
            return avg_cost + 1e6 * avg_err + 1e5 * max_err
    else:
        TEST_DISTRIBUTIONS = [
            {"dist": "lognormal", "mu": 0.7, "sigma": 0.13},
            {"dist": "normal", "mu": 0.0, "sigma": 1.0},
            {"dist": "exponential", "lam": 1.0},
            {"dist": "pareto", "shape": 2.5, "scale": 1.0},
            {"dist": "student_t", "df": 3.0, "loc": 0.0, "scale": 1.0},
            {"dist": "skew_normal", "skew": 4.0, "loc": 0.0, "scale": 1.0},
            {"dist": "beta", "a": 2.0, "b": 5.0},
            {"dist": "mixture"},
        ]

        dist_data = []
        for d in TEST_DISTRIBUTIONS:
            gp, pr = get_distribution(
                d["dist"],
                2 ** args.num_qubits,
                mu=d.get("mu", 0.0),
                sigma=d.get("sigma", 1.0),
                lam=d.get("lam", 1.0),
                shape=d.get("shape", 2.5),
                scale=d.get("scale", 1.0),
                a=d.get("a", 2.0),
                b=d.get("b", 5.0),
                df=d.get("df", 3.0),
                skew=d.get("skew", 4.0),
                loc=d.get("loc", 0.0),
            )
            cdf = np.cumsum(pr)
            ref_idx = int(np.searchsorted(cdf, args.alpha, side="left"))
            ref_var = float(gp[ref_idx])
            dist_data.append({
                "name": d["dist"],
                "grid": gp,
                "probs": pr,
                "ref_idx": ref_idx,
                "ref_var": ref_var,
            })

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=20)

        def objective(trial: optuna.Trial):
            # Quantum hyperparameters
            epsilon = trial.suggest_float("epsilon", 0.01, 0.15)
            alpha_fail = trial.suggest_float("alpha_fail", 0.001, 0.05, log=True)
            bound = trial.suggest_float("bound", 0.0, 0.15, log=True)
            max_width = trial.suggest_int("max_width", 20, 40, log=True)
            max_depth = trial.suggest_int("max_depth", 500, 5000, log=True)
            prob_tol_mult = trial.suggest_float("prob_tol_mult", 0.05, 0.3)
            initial_frac = trial.suggest_float("initial_frac", 0.1, 0.5)
            adaptive_eps = trial.suggest_categorical("adaptive_eps", [True, False])

            if adaptive_eps:
                eps_start = trial.suggest_float("eps_start", epsilon, epsilon * 4)
                eps_end = trial.suggest_float("eps_end", epsilon * 0.5, epsilon)
            else:
                eps_start = None
                eps_end = None

            NUM_SEEDS = 1

            total_cost = 0.0
            total_err = 0.0
            max_err = 0.0

            for i, dd in enumerate(dist_data):
                dist_cost = 0.0
                dist_err = 0.0

                for seed_offset in range(NUM_SEEDS):
                    try:
                        res = estimate_var_quantum(
                            probs=dd["probs"],
                            grid_points=dd["grid"],
                            alpha=args.alpha,
                            epsilon=epsilon,
                            alpha_fail=alpha_fail,
                            bound=bound,
                            max_width=max_width,
                            max_depth=max_depth,
                            prob_tol=args.alpha * prob_tol_mult,
                            initial_frac=initial_frac,
                            adaptive_eps=adaptive_eps,
                            eps_start=eps_start,
                            eps_end=eps_end,
                            max_steps=args.max_steps,
                        )

                        var = res.var
                        cost = res.total_oracle_calls
                    except Exception as e:
                        print(f"Quantum trial failed: {e}")
                        return float("inf")

                    err = abs(var - dd["ref_var"]) / (abs(dd["ref_var"]) + 1e-9)
                    dist_cost += cost
                    dist_err += err

                dist_cost /= NUM_SEEDS
                dist_err /= NUM_SEEDS

                total_cost += dist_cost
                total_err += dist_err
                max_err = max(max_err, dist_err)

                trial.report(total_cost / (i + 1), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            avg_cost = total_cost / len(dist_data)
            avg_err = total_err / len(dist_data)
            return avg_cost + 1e6 * avg_err + 1e5 * max_err

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    best = {
        "mode": "classical_robust",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "num_distributions": len(dist_data),
        "distributions": [d["name"] for d in dist_data],
        "alpha": args.alpha,
        "num_qubits": args.num_qubits,
    }
    print(json.dumps(best, indent=2))

    with open(args.out, "w") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
# {
#   "mode": "classical_robust",
#   "best_value": 13884.984625977311,
#   "best_params": {
#     "budget": 502,
#     "confidence": 0.9790934333417373,
#     "method": "is_qmc",
#     "tilt_tau": 0.11372799886830216,
#     "use_control_variate": false
#   },
#   "num_distributions": 8,
#   "distributions": [
#     "lognormal",
#     "normal",
#     "exponential",
#     "pareto",
#     "student_t",
#     "skew_normal",
#     "beta",
#     "mixture"
#   ],
#   "alpha": 0.05,
#   "num_qubits": 7
# }