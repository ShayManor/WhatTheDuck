#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

import numpy as np
import optuna
import scipy.stats as st

TailMode = Literal["pnl_leq", "loss_geq"]


@dataclass(frozen=True)
class TailQuery:
    alpha_target: float
    tail_mode: TailMode
    threshold_value: Optional[float] = None
    index: Optional[int] = None
    grid_points: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ProbEstimate:
    p_hat: float
    ci_low: float
    ci_high: float
    cost: int
    meta: Dict[str, Any]


# ----------- Helpers -----------

def get_log_normal_probabilities(mu_normal, sigma_normal, num_points):
    mean = np.exp(mu_normal + sigma_normal ** 2 / 2)
    var = (np.exp(sigma_normal ** 2) - 1) * np.exp(2 * mu_normal + sigma_normal ** 2)
    std = np.sqrt(var)
    low = max(0.0, mean - 3 * std)
    high = mean + 3 * std
    x = np.linspace(low, high, num_points)
    pdf = st.lognorm.pdf(x, s=sigma_normal, scale=np.exp(mu_normal))
    p = pdf / np.sum(pdf)
    return x, p


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


# Classical GPU estimator (Monte Carlo on discrete probs)

class ClassicalDiscreteMC:
    def __init__(self, probs: np.ndarray):
        self.p = probs.astype(np.float64)
        self.K = len(self.p)
        try:
            import torch
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.p_t = torch.tensor(self.p, device=self.device, dtype=torch.float32)
        except Exception:
            self.torch = None
            self.device = "cpu"
            self.p_t = None

    def estimate_tail_prob(self, query: TailQuery, *, budget: int, confidence: float = 0.99, seed: Optional[int] = None,
                           **_):
        if query.index is None:
            raise ValueError("query.index required")
        n = int(budget)
        idx = int(query.index)

        if self.torch is not None:
            g = self.torch.Generator(device=self.device)
            if seed is not None:
                g.manual_seed(int(seed))
            # torch.multinomial draws indices ~ categorical(p) efficiently on GPU
            samples = self.torch.multinomial(self.p_t, n, replacement=True, generator=g)
            k = int((samples < idx).sum().item())
        else:
            rng = np.random.default_rng(seed)
            samples = rng.choice(np.arange(self.K), size=n, p=self.p)
            k = int(np.sum(samples < idx))

        p_hat = k / n
        lo, hi = wilson_ci(k, n, confidence)
        return ProbEstimate(p_hat=float(p_hat), ci_low=lo, ci_high=hi, cost=n,
                            meta={"device": self.device, "k": k, "n": n})


# Quantum estimator (Classiq IQAE)

class QuantumIQAECDF:
    def __init__(self, probs_list, num_qubits: int, max_width: int = 28):
        from classiq import qfunc, qperm, QArray, QBit, QNum, Const, inplace_prepare_state, Constraints, Preferences
        from classiq.applications.iqae.iqae import IQAE

        self.IQAE = IQAE
        self.Constraints = Constraints
        self.Preferences = Preferences
        self.num_qubits = int(num_qubits)
        self.max_width = int(max_width)

        # Bind probs into closure for Classiq decorators
        self.PROBS = list(map(float, probs_list))
        self.GLOBAL_INDEX = 0

        @qfunc(synthesize_separately=True)
        def state_preparation(asset: QArray[QBit], ind: QBit):
            inplace_prepare_state(self.PROBS, bound=0, target=asset)
            payoff(asset=asset, ind=ind)

        @qperm
        def payoff(asset: Const[QNum], ind: QBit):
            ind ^= asset < self.GLOBAL_INDEX

        self.state_preparation = state_preparation

    def _cost_from_iqae(self, res) -> int:
        # Best-effort: sum shots*(2k+1) if iterations_data exists; else 0
        iters = getattr(res, "iterations_data", None)
        if not iters:
            return 0
        total = 0
        for it in iters:
            k = int(getattr(it, "grover_iterations", 0))
            sr = getattr(it, "sample_results", None)
            shots = int(getattr(sr, "num_shots", 0)) if sr is not None else 0
            total += shots * (2 * k + 1)
        return int(total)

    def estimate_tail_prob(self, query: TailQuery, *, epsilon: float, alpha: float,
                           max_total_queries: Optional[int] = None, seed: Optional[int] = None, **_):
        if query.index is None:
            raise ValueError("query.index required")
        self.GLOBAL_INDEX = int(query.index)

        iqae = self.IQAE(
            state_prep_op=self.state_preparation,
            problem_vars_size=self.num_qubits,
            constraints=self.Constraints(max_width=self.max_width),
            preferences=self.Preferences(machine_precision=self.num_qubits),
        )
        res = iqae.run(epsilon=float(epsilon), alpha=float(alpha))
        p_hat = float(res.estimation)
        ci = list(res.confidence_interval)
        cost = self._cost_from_iqae(res)
        if max_total_queries is not None:
            cost = min(cost, int(max_total_queries))
        return ProbEstimate(p_hat=p_hat, ci_low=float(ci[0]), ci_high=float(ci[1]), cost=int(cost),
                            meta={"epsilon": epsilon, "alpha": alpha, "seed": seed})



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-in", action="store_true")
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--quantum", action="store_true")
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
    args = ap.parse_args()

    if args.log_in:
        import classiq
        classiq.authenticate()

    if args.classical == args.quantum:
        raise SystemExit("Pick exactly one: --classical or --quantum")

    grid_points, probs = get_log_normal_probabilities(args.mu, args.sigma, 2 ** args.num_qubits)
    cdf = np.cumsum(probs)
    ref_idx = int(np.searchsorted(cdf, args.alpha, side="left"))
    ref_var = float(grid_points[ref_idx])

    prob_tol = args.prob_tol if args.prob_tol is not None else args.alpha / 10.0

    # Estimator selection
    if args.classical:
        est = ClassicalDiscreteMC(probs)

        def estimator(q: TailQuery, **kw):
            return est.estimate_tail_prob(q, **kw)

        sampler = optuna.samplers.TPESampler(seed=args.seed)  # Bayesian TPE
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8)

        def objective(trial: optuna.Trial):
            budget = trial.suggest_int("budget", 500, 200_000, log=True)
            conf = trial.suggest_float("confidence", 0.90, 0.999)
            var, vidx, cost, _ = solve_var_bisect(
                estimator,
                alpha_target=args.alpha,
                tail_mode="pnl_leq",
                grid_points=grid_points,
                lo_index=0,
                hi_index=len(grid_points) - 1,
                prob_tol=prob_tol,
                value_tol=args.value_tol,
                max_steps=args.max_steps,
                est_params={"budget": budget, "confidence": conf, "seed": args.seed + trial.number},
            )
            err = abs(var - ref_var)
            trial.report(err, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            # Penalize violations if tau > 0; else just minimize cost for frontier building later
            return cost + (1e9 * max(0.0, err - args.tau)) if args.tau > 0 else cost + 1e6 * err

    else:
        est = QuantumIQAECDF(probs, num_qubits=args.num_qubits)

        def estimator(q: TailQuery, **kw):
            return est.estimate_tail_prob(q, **kw)

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8)

        def objective(trial: optuna.Trial):
            epsilon = trial.suggest_float("epsilon", 0.01, 0.2, log=True)
            alpha_fail = trial.suggest_float("alpha_fail", 1e-3, 0.1, log=True)
            var, vidx, cost, _ = solve_var_bisect(
                estimator,
                alpha_target=args.alpha,
                tail_mode="pnl_leq",
                grid_points=grid_points,
                lo_index=0,
                hi_index=len(grid_points) - 1,
                prob_tol=prob_tol,
                value_tol=args.value_tol,
                max_steps=args.max_steps,
                est_params={"epsilon": epsilon, "alpha": alpha_fail, "seed": args.seed + trial.number},
            )
            err = abs(var - ref_var)
            trial.report(err, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return cost + (1e9 * max(0.0, err - args.tau)) if args.tau > 0 else cost + 1e6 * err

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    best = {
        "mode": "classical" if args.classical else "quantum",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "ref_var": ref_var,
        "ref_idx": ref_idx,
        "num_qubits": args.num_qubits,
        "alpha": args.alpha,
    }
    print(json.dumps(best, indent=2))

    with open(args.out, "w") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
