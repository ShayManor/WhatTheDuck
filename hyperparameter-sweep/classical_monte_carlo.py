import numpy as np

from sweep import ClassicalDiscreteMC, solve_var_bisect, TailQuery, get_distribution


def estimate_var_classical(
        probs: np.ndarray,
        grid_points: np.ndarray,
        alpha: float = 0.05,
        seed: int = 42,
        # Best hyperparameters from sweep
        budget: int = 502,
        confidence: float = 0.979,
        method: str = "is_qmc",
        tilt_tau: float = 0.114,
        use_control_variate: bool = False,
        prob_tol: float = None,
        max_steps: int = 64,
) -> dict:
    """
    Estimate VaR using optimized classical Monte Carlo.

    Args:
        probs: Discretized probability distribution
        grid_points: Corresponding grid values
        alpha: Target tail probability (e.g., 0.05 for 5% VaR)
        seed: Random seed
        budget: Samples per bisection step
        confidence: CI confidence level
        method: Sampling method ("is_qmc", "plain", "stratified", etc.)
        tilt_tau: IS exponential tilting parameter
        use_control_variate: Whether to use control variates
        prob_tol: CI ambiguity tolerance (default: alpha/10)
        max_steps: Max bisection iterations

    Returns:
        dict with keys:
            - var: Estimated VaR value
            - var_index: Grid index of VaR
            - total_samples: Total samples used (the cost)
            - bisection_steps: Number of bisection iterations
            - trace: List of (index, p_hat, ci_low, ci_high, cost) per step
    """
    if prob_tol is None:
        prob_tol = alpha / 10.0

    estimator = ClassicalDiscreteMC(probs)

    def est_fn(q: TailQuery, **kw):
        return estimator.estimate_tail_prob(q, **kw)

    var, var_idx, total_cost, trace = solve_var_bisect(
        est_fn,
        alpha_target=alpha,
        tail_mode="pnl_leq",
        grid_points=grid_points,
        lo_index=0,
        hi_index=len(grid_points) - 1,
        prob_tol=prob_tol,
        value_tol=0.0,
        max_steps=max_steps,
        est_params={
            "budget": budget,
            "confidence": confidence,
            "seed": seed,
            "method": method,
            "tilt_tau": tilt_tau,
            "use_control_variate": use_control_variate,
        },
    )

    return {
        "var": var,
        "var_index": var_idx,
        "total_samples": total_cost,
        "bisection_steps": len(trace),
        "trace": trace,
    }

if __name__ == '__main__':
    import scipy.stats as st
    grid, probs = get_distribution("pareto", num_points=512, shape=2.5, scale=1.0)
    true_var = st.pareto.ppf(0.05, b=2.5, scale=1.0)
    result = estimate_var_classical(probs, grid, alpha=0.05, seed=123)

    print(f"VaR estimate: {result['var']:.4f}")
    print(f"Total samples: {result['total_samples']}")
    print(f"Bisection steps: {result['bisection_steps']}")
    print(f"True VaR: {true_var:.4f}")
