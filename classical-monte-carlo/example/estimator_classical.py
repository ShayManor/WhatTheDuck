# estimators_classical.py
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from structures import TailQuery, ProbEstimate, ClassicalProbEstimator


def _normal_approx_z(confidence: float) -> float:
    """
    Two-sided z for confidence level (e.g., 0.99 -> z ~ 2.575).
    Uses a dependency-free approximation (Abramowitz-Stegun style).
    """
    delta = 1.0 - float(confidence)
    p = 1.0 - delta / 2.0

    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    ) / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)


def _wilson_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = _normal_approx_z(confidence)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


# ======================================================================================
# Baseline: Discrete Monte Carlo (your original estimator)
# ======================================================================================


class DiscreteMonteCarloEstimator(ClassicalProbEstimator):
    """
    Estimates p = P(sample_index < query.index) by sampling from the discrete distribution `probs`.

    Cost accounting:
      - cost = number of samples used (<= budget)
    """

    def __init__(self, probs: list[float]):
        self._probs = np.array(probs, dtype=float)
        s = float(np.sum(self._probs))
        if s <= 0:
            raise ValueError("probs must sum to > 0")
        self._probs = self._probs / s
        self._support = np.arange(len(probs), dtype=int)

    def estimate_tail_prob(
        self,
        query: TailQuery,
        *,
        budget: int,
        confidence: float = 0.99,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> ProbEstimate:
        if query.index is None:
            raise ValueError("DiscreteMonteCarloEstimator requires query.index")
        n = int(budget)
        if n <= 0:
            raise ValueError("budget must be positive")

        rng = np.random.default_rng(seed)
        samples = rng.choice(self._support, size=n, p=self._probs)

        # Tail event consistent with the notebook: CDF(index) = P(bin < index)
        successes = int(np.sum(samples < int(query.index)))
        p_hat = successes / n
        ci_low, ci_high = _wilson_ci(successes, n, confidence)

        return ProbEstimate(
            p_hat=float(p_hat),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            cost=int(n),
            meta={
                "method": "discrete_mc",
                "seed": seed,
                "successes": successes,
                "n": n,
                "confidence": confidence,
            },
        )


# ======================================================================================
# Advanced: IS + stratification + QMC + control variates + safe sample reuse (Option B)
# ======================================================================================


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
    """
    1D Halton points for indices i in [start_index_1based, start_index_1based + n - 1]
    Returned values are in [0,1).
    If scramble_seed is not None, we scramble digits with a fixed permutation.
    """
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


class AdvancedDiscreteMonteCarloEstimator(ClassicalProbEstimator):
    """
    Drop-in replacement for DiscreteMonteCarloEstimator with:
      - IS (exponential tilting), stratification, QMC, control variates
      - Option B caching: cached sample set grows to satisfy larger budgets safely
      - cost returned is NEW samples generated in the call

    Tail event is P(bin < query.index), consistent with your baseline and solve_var.
    """

    def __init__(self, probs: list[float]):
        self._p = _validate_pmf(np.array(probs, dtype=float))
        self._support = np.arange(self._p.size, dtype=int)
        self._cdf_p = _cdf_from_pmf(self._p)

        # Control variate: X = index has known expectation under p
        self._EX_index = float(np.sum(self._support * self._p))

        # Cache maps key -> dict holding samples and state for safe extension
        self._cache: Dict[_CacheKey, Dict[str, Any]] = {}

    def _proposal_from_tilt(self, tau: float) -> np.ndarray:
        """
        Exponential tilt that increases probability mass on small indices (left tail).
        q_i ∝ p_i * exp(-tau * i), tau >= 0.
        """
        tau = float(tau)
        if tau < 0:
            raise ValueError("tilt_tau must be >= 0")
        logw = -tau * self._support.astype(float)
        q_unnorm = self._p * np.exp(logw - np.max(logw))
        return _validate_pmf(q_unnorm)

    def _make_strata(self, num_strata: int) -> list[Tuple[int, int, float]]:
        """
        Make contiguous strata of indices with approximately equal probability mass under p.
        Returns list of (lo, hi_exclusive, mass).
        """
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

        # Continue Halton sequence per base using counters
        used = int(qmc_counters.get(base, 0))
        start = used + 1  # 1-based index
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
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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

        # No stratification
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

        # Stratification by mass under the true pmf p
        S = self._make_strata(strata)
        masses = np.array([m for _, _, m in S], dtype=float)
        masses = masses / float(np.sum(masses))

        # Stable mass allocation (pilot unused for allocation here, but kept for API compatibility)
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

            base = 2 + (s % 5)  # small decorrelation across strata
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
    ) -> Tuple[float, float, float, float, Dict[str, Any]]:
        n = int(idx.size)
        if n <= 0:
            return 0.0, 0.0, 1.0, 0.0, {"cv_used": False}

        y = (idx < int(index_thr)).astype(float)
        w = np.asarray(w, dtype=float)

        sum_w = float(np.sum(w))
        if sum_w <= 0:
            raise ValueError("sum of weights must be > 0")

        cv_meta: Dict[str, Any]
        if use_cv:
            # Control variate using X = index with known E_p[X]
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

        # Self-normalized importance sampling estimate
        phat = float(np.sum(w * y) / sum_w)

        # Effective sample size for CI approximation
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
        """
        Compatible with solve_var.

        Supported kwargs (optional):
          - method: "plain" | "is" | "stratified" | "is_stratified"
                    plus qmc variants: "qmc", "is_qmc", "stratified_qmc", "is_stratified_qmc"
            Default: "plain"
          - tilt_tau: float >= 0. Default: 0.0 (only used for IS methods)
          - qmc: bool. Default: inferred from method name containing "qmc"
          - scramble_seed: int | None. Default: None
          - strata: int >= 1. Default: 1
          - pilot: int >= 0. Default: 0 (kept for API compatibility)
          - use_control_variate: bool. Default: False

          - reuse_id: str | None. If provided, enables Option B caching across calls:
              cache grows to satisfy larger budgets safely and stores RNG/QMC state.
          - batch_size: int. Default: 8192 (when generating new samples)

          - target_prob: float | None. If set, enables estimator-level adaptive stopping:
              stop when CI entirely above or below target_prob.
          - ci_width_tol: float | None. If set, stop when (ci_high-ci_low) <= tol.
        """
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

        # Load cache if present
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

        # Option B: grow cache to satisfy larger budgets safely
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

                # Optional adaptive stopping inside estimator
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

        # Store updated cache
        if cache_key is not None:
            self._cache[cache_key] = {
                "idx": idx,
                "w": w,
                "rng_state": rng_state,
                "qmc_counters": dict(qmc_counters),
                "meta": dict(base_meta),
            }

        # Respect budget: estimate using exactly first n_req samples
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


__all__ = [
    "DiscreteMonteCarloEstimator",
    "AdvancedDiscreteMonteCarloEstimator",
]
