# example_sweep_call.py
import numpy as np
import scipy

from classiq import Constraints, Preferences

from estimators_classical import (
    DiscreteMonteCarloEstimator,
    AdvancedDiscreteMonteCarloEstimator,
)
from estimators_quantum import ClassiqIQAECDFEstimator
from var_search import solve_var

# Same variable names as the notebook
num_qubits = 7
mu = 0.7
sigma = 0.13
ALPHA = 0.07
TOLERANCE = ALPHA / 10


def get_log_normal_probabilities(mu_normal, sigma_normal, num_points):
    log_normal_mean = np.exp(mu_normal + sigma_normal**2 / 2)
    log_normal_variance = (np.exp(sigma_normal**2) - 1) * np.exp(
        2 * mu_normal + sigma_normal**2
    )
    log_normal_stddev = np.sqrt(log_normal_variance)
    low = np.maximum(0, log_normal_mean - 3 * log_normal_stddev)
    high = log_normal_mean + 3 * log_normal_stddev
    x = np.linspace(low, high, num_points)
    return x, scipy.stats.lognorm.pdf(x, s=sigma_normal, scale=np.exp(mu_normal))


grid_points, probs = get_log_normal_probabilities(mu, sigma, 2**num_qubits)
probs = (probs / np.sum(probs)).tolist()

# Reference VaR on discretized grid (exact)
cdf = np.cumsum(probs)
VAR_index_ref = int(np.searchsorted(cdf, ALPHA, side="left"))
VAR = float(grid_points[VAR_index_ref])
print(f"[Reference discrete] VAR_index={VAR_index_ref}, VAR={VAR}")

# Common bracket for index search
lo_index = 0
hi_index = len(grid_points) - 1

# ---------------------------------------
# Classical sweep examples (baseline vs advanced)
# ---------------------------------------
baseline_est = DiscreteMonteCarloEstimator(probs=probs)
advanced_est = AdvancedDiscreteMonteCarloEstimator(probs=probs)

for budget in [500, 2_000, 10_000]:
    # Baseline
    res_base = solve_var(
        baseline_est.estimate_tail_prob,
        alpha_target=ALPHA,
        tail_mode="pnl_leq",
        grid_points=grid_points,
        lo_index=lo_index,
        hi_index=hi_index,
        value_tol=0.0,          # index-based stopping dominates; set nonzero if desired
        prob_tol=TOLERANCE,     # same meaning as notebook tolerance in probability space
        max_steps=64,
        estimator_params={
            "budget": budget,
            "confidence": 0.99,
            "seed": 123,
            # optional refinement knobs (also sweepable)
            "max_refinements": 1,
            "refine_mult": 2.0,
        },
    )
    err_base = abs(res_base.var_value - VAR)
    print(
        f"[Classical MC Baseline] budget={budget:>6}  var={res_base.var_value:.6f}  "
        f"err={err_base:.6f}  cost={res_base.total_cost}"
    )

    # Advanced
    res_adv = solve_var(
        advanced_est.estimate_tail_prob,
        alpha_target=ALPHA,
        tail_mode="pnl_leq",
        grid_points=grid_points,
        lo_index=lo_index,
        hi_index=hi_index,
        value_tol=0.0,
        prob_tol=TOLERANCE,
        max_steps=64,
        estimator_params={
            "budget": budget,
            "confidence": 0.99,
            "seed": 123,

            # Advanced knobs
            "method": "is_stratified_qmc",   # try also: "plain", "is", "stratified", "is_stratified"
            "tilt_tau": 0.02,               # importance sampling tilt, tune this
            "strata": 16,                   # number of strata
            "qmc": True,                    # Halton-based QMC
            "scramble_seed": 7,             # makes QMC less brittle
            "use_control_variate": True,    # control variate on index

            # Safe reuse across thresholds and refinements (Option B cache growth)
            "reuse_id": f"adv_budget_{budget}",

            # optional estimator-level adaptive stopping (solve_var already refines too)
            # "target_prob": ALPHA,
            # "ci_width_tol": TOLERANCE,
            # "batch_size": 50_000,

            # refinement knobs (also sweepable)
            "max_refinements": 1,
            "refine_mult": 2.0,
        },
    )
    err_adv = abs(res_adv.var_value - VAR)
    print(
        f"[Classical MC Advanced] budget={budget:>6}  var={res_adv.var_value:.6f}  "
        f"err={err_adv:.6f}  cost={res_adv.total_cost}"
    )

# ---------------------------------------
# Quantum sweep example (epsilon/alpha sweep)
# ---------------------------------------
quantum_est = ClassiqIQAECDFEstimator(
    probs=probs,
    num_qubits=num_qubits,
    constraints=Constraints(max_width=28),
    preferences=Preferences(machine_precision=num_qubits),
    cost_mode="A_calls",  # or "shots", or "AA_calls"
)

for epsilon, alpha in [(0.10, 0.05), (0.05, 0.01), (0.02, 0.01)]:
    res = solve_var(
        quantum_est.estimate_tail_prob,
        alpha_target=ALPHA,
        tail_mode="pnl_leq",
        grid_points=grid_points,
        lo_index=lo_index,
        hi_index=hi_index,
        value_tol=0.0,
        prob_tol=TOLERANCE,
        max_steps=64,
        estimator_params={
            "epsilon": epsilon,
            "alpha": alpha,
            "max_total_queries": None,
            # optional refinement knobs (also sweepable)
            "max_refinements": 1,
            "refine_mult": 2.0,  # decreases epsilon when ambiguous
        },
    )
    err = abs(res.var_value - VAR)
    print(
        f"[Quantum IQAE] eps={epsilon:<5} alpha={alpha:<5}  var={res.var_value:.6f}  "
        f"err={err:.6f}  cost={res.total_cost}"
    )
