#!/usr/bin/env python3
"""
Quantum IQAE VaR Estimator - Tunable Version

Usage:
    from quantum_var import estimate_var_quantum

    result = estimate_var_quantum(
        probs=probs,
        grid_points=grid_points,
        alpha=0.05,
        epsilon=0.05,
    )
    print(f"VaR: {result['var']}, Queries: {result['total_oracle_calls']}")
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any


@dataclass
class QuantumVaRResult:
    var: float
    var_index: int
    total_oracle_calls: int
    bisection_steps: int
    epsilon: float
    alpha_fail: float
    confidence: float
    trace: List[Tuple[int, float, float, float, int, float]]  # (idx, p_hat, ci_lo, ci_hi, cost, eps_used)


def estimate_var_quantum(
        probs: np.ndarray,
        grid_points: np.ndarray,
        alpha: float = 0.05,
        epsilon: float = 0.05,
        alpha_fail: float = 0.01,
        # === TUNABLE HYPERPARAMETERS ===
        bound: float = 0.0,  # State prep approximation (0 = exact, higher = shallower circuit)
        max_width: int = 28,  # Circuit width constraint
        max_depth: Optional[int] = None,  # Circuit depth constraint (None = unconstrained)
        machine_precision: Optional[int] = None,  # Numerical precision (default = num_qubits)
        prob_tol: Optional[float] = None,  # Bisection CI tolerance (default = alpha/10)
        initial_frac: Optional[float] = None,  # Starting index as fraction of grid (default = 0.25)
        adaptive_eps: bool = False,  # Use adaptive epsilon schedule
        eps_start: Optional[float] = None,  # Starting epsilon (if adaptive)
        eps_end: Optional[float] = None,  # Ending epsilon (if adaptive)
        max_steps: int = 64,
) -> QuantumVaRResult:
    """
    Estimate VaR using Quantum IQAE with tunable hyperparameters.

    Args:
        probs: Discretized probability distribution (will be normalized)
        grid_points: Corresponding grid values
        alpha: Target tail probability (e.g., 0.05 for 5% VaR)
        epsilon: IQAE precision (|p_hat - p| <= epsilon with high prob)
        alpha_fail: IQAE failure probability (confidence = 1 - alpha_fail)

        # Tunable hyperparameters:
        bound: State preparation approximation bound (0 = exact)
        max_width: Maximum circuit width
        max_depth: Maximum circuit depth (None = no limit)
        machine_precision: Numerical precision in qubits
        prob_tol: Bisection decision tolerance
        initial_frac: Starting index as fraction of grid size
        adaptive_eps: Whether to use adaptive epsilon schedule
        eps_start: Starting epsilon for adaptive schedule
        eps_end: Ending epsilon for adaptive schedule
        max_steps: Maximum bisection iterations

    Returns:
        QuantumVaRResult with var, total_oracle_calls, and trace
    """
    from classiq import Constraints, Preferences, qfunc, qperm, QArray, QBit, QNum, Const, inplace_prepare_state
    from classiq.applications.iqae.iqae import IQAE

    # Normalize and convert probs
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr = probs_arr / np.sum(probs_arr)
    probs_list = probs_arr.tolist()

    num_qubits = int(np.round(np.log2(len(probs_list))))
    assert 2 ** num_qubits == len(probs_list), "probs length must be power of 2"

    # Set defaults
    if prob_tol is None:
        prob_tol = alpha / 10.0
    if machine_precision is None:
        machine_precision = num_qubits
    if initial_frac is None:
        initial_frac = 0.25
    if adaptive_eps:
        if eps_start is None:
            eps_start = epsilon * 2
        if eps_end is None:
            eps_end = epsilon

    # Mutable state for threshold (needed for closure)
    state = {"threshold_index": 0}

    # Define quantum functions with tunable bound
    @qfunc(synthesize_separately=True)
    def state_preparation(asset: QArray[QBit], ind: QBit):
        load_distribution(asset=asset)
        payoff(asset=asset, ind=ind)

    @qfunc
    def load_distribution(asset: QNum):
        inplace_prepare_state(probs_list, bound=bound, target=asset)

    @qperm
    def payoff(asset: Const[QNum], ind: QBit):
        ind ^= asset < state["threshold_index"]

    def run_iqae_at_index(index: int, eps: float) -> Tuple[float, float, float, int]:
        """
        Run IQAE for given threshold index and epsilon.
        Returns: (p_hat, ci_low, ci_high, oracle_calls)
        """
        state["threshold_index"] = int(index)

        # Build constraints
        constraints_kwargs = {"max_width": max_width}
        if max_depth is not None:
            constraints_kwargs["max_depth"] = max_depth

        iqae = IQAE(
            state_prep_op=state_preparation,
            problem_vars_size=num_qubits,
            constraints=Constraints(**constraints_kwargs),
            preferences=Preferences(machine_precision=machine_precision),
        )

        res = iqae.run(epsilon=eps, alpha=alpha_fail)

        # Count oracle calls: each shot with k Grover iterations uses (2k+1) oracle calls
        oracle_calls = 0
        iterations_data = getattr(res, "iterations_data", []) or []
        for it in iterations_data:
            k = getattr(it, "grover_iterations", 0) or 0
            shots = 0
            if hasattr(it, "sample_results") and it.sample_results is not None:
                shots = getattr(it.sample_results, "num_shots", 0) or 0
            elif hasattr(it, "num_shots"):
                shots = getattr(it, "num_shots", 0) or 0
            oracle_calls += shots * (2 * k + 1)

        ci_low, ci_high = res.confidence_interval
        return float(res.estimation), float(ci_low), float(ci_high), int(oracle_calls)

    # Binary search
    n_points = len(grid_points)
    lo, hi = 0, n_points - 1
    initial_index = int(n_points * initial_frac)

    total_cost = 0
    trace = []

    for step in range(max_steps):
        if lo >= hi or (grid_points[hi] - grid_points[lo]) <= 0:
            break

        mid = (lo + hi) // 2

        # Compute epsilon for this step
        if adaptive_eps:
            progress = step / max(1, max_steps - 1)
            current_eps = eps_start * (1 - progress) + eps_end * progress
        else:
            current_eps = epsilon

        p_hat, ci_low, ci_high, cost = run_iqae_at_index(mid, current_eps)
        total_cost += cost
        trace.append((mid, p_hat, ci_low, ci_high, cost, current_eps))

        # CI-driven bisection decision
        if alpha > ci_high + prob_tol:
            lo = mid + 1
        elif alpha < ci_low - prob_tol:
            hi = mid
        else:
            # Ambiguous: fallback to point estimate
            if p_hat < alpha:
                lo = mid + 1
            else:
                hi = mid

    return QuantumVaRResult(
        var=float(grid_points[lo]),
        var_index=int(lo),
        total_oracle_calls=total_cost,
        bisection_steps=len(trace),
        epsilon=epsilon,
        alpha_fail=alpha_fail,
        confidence=1.0 - alpha_fail,
        trace=trace,
    )


def quantum_var_objective(
        trial,
        probs: np.ndarray,
        grid_points: np.ndarray,
        alpha: float,
        ref_var: float,
        epsilon: float = 0.05,
        alpha_fail: float = 0.01,
) -> float:
    """
    Optuna objective for tuning quantum hyperparameters.

    Usage:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: quantum_var_objective(t, probs, grid, alpha, ref_var),
            n_trials=100
        )
    """
    # Suggest hyperparameters
    bound = trial.suggest_float("bound", 0.0, 0.15)
    max_width = trial.suggest_int("max_width", 20, 40)
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

    try:
        result = estimate_var_quantum(
            probs=probs,
            grid_points=grid_points,
            alpha=alpha,
            epsilon=epsilon,
            alpha_fail=alpha_fail,
            bound=bound,
            max_width=max_width,
            max_depth=max_depth,
            prob_tol=alpha * prob_tol_mult,
            initial_frac=initial_frac,
            adaptive_eps=adaptive_eps,
            eps_start=eps_start,
            eps_end=eps_end,
        )

        # Objective: minimize cost + penalize error
        err = abs(result.var - ref_var) / (abs(ref_var) + 1e-9)
        cost = result.total_oracle_calls

        return cost + 1e6 * err

    except Exception as e:
        # If synthesis fails, return high cost
        print(f"Trial failed: {e}")
        return float("inf")


# ============================================================================
# Convenience functions for direct comparison with classical
# ============================================================================

def get_distribution(dist_name: str, num_points: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Factory function for test distributions."""
    import scipy.stats as st

    if dist_name == "lognormal":
        mu = kwargs.get("mu", 0.7)
        sigma = kwargs.get("sigma", 0.13)
        mean = np.exp(mu + sigma ** 2 / 2)
        var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        std = np.sqrt(var)
        low = max(0.0, mean - 3 * std)
        high = mean + 3 * std
        x = np.linspace(low, high, num_points)
        pdf = st.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    elif dist_name == "normal":
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, num_points)
        pdf = st.norm.pdf(x, loc=mu, scale=sigma)
    elif dist_name == "pareto":
        shape = kwargs.get("shape", 2.5)
        scale = kwargs.get("scale", 1.0)
        high = st.pareto.ppf(0.99, b=shape, scale=scale)
        x = np.linspace(scale, high, num_points)
        pdf = st.pareto.pdf(x, b=shape, scale=scale)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

    p = pdf / np.sum(pdf)
    return x, p


def compute_true_var(probs: np.ndarray, grid_points: np.ndarray, alpha: float) -> Tuple[float, int]:
    """Compute exact discrete VaR."""
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, alpha, side="left"))
    return float(grid_points[idx]), idx


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantum VaR Estimation")
    parser.add_argument("--dist", type=str, default="lognormal")
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--alpha-fail", type=float, default=0.01)
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter sweep")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    # Setup
    num_points = 2 ** args.num_qubits
    grid_points, probs = get_distribution(args.dist, num_points)
    true_var, true_idx = compute_true_var(probs, grid_points, args.alpha)

    print(f"Distribution: {args.dist}")
    print(f"Grid points: {num_points}")
    print(f"True VaR at α={args.alpha}: {true_var:.4f} (index {true_idx})")
    print()

    if args.tune:
        import optuna

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: quantum_var_objective(
                t, probs, grid_points, args.alpha, true_var,
                epsilon=args.epsilon, alpha_fail=args.alpha_fail
            ),
            n_trials=args.trials,
        )

        print("\n=== Best Hyperparameters ===")
        print(f"Best value: {study.best_value:.2f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
    else:
        # Single run with defaults
        result = estimate_var_quantum(
            probs=probs,
            grid_points=grid_points,
            alpha=args.alpha,
            epsilon=args.epsilon,
            alpha_fail=args.alpha_fail,
        )

        print(f"=== Quantum VaR Result ===")
        print(f"VaR estimate: {result.var:.4f}")
        print(f"True VaR:     {true_var:.4f}")
        print(f"Error:        {abs(result.var - true_var):.4f}")
        print()
        print(f"Total oracle calls: {result.total_oracle_calls}")
        print(f"Bisection steps:    {result.bisection_steps}")
        print(f"Epsilon:            {result.epsilon}")
        print(f"Confidence:         {result.confidence:.1%}")

if __name__ == '__main__':
    grid, probs = get_distribution("lognormal", 128, mu=0.7, sigma=0.13)
    true_var, _ = compute_true_var(probs, grid, alpha=0.05)

    result = estimate_var_quantum(
        probs=probs,
        grid_points=grid,
        alpha=0.05,
        epsilon=0.05,
        # Tunable params:
        bound=0.01,
        adaptive_eps=True,
        eps_start=0.15,
        eps_end=0.03,
    )

    print(f"VaR: {result.var:.4f}, Queries: {result.total_oracle_calls}")