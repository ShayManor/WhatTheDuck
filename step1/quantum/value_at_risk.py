"""
Value at Risk (VaR) with Iterative Quantum Amplitude Estimation (IQAE)

This script performs a parameter sweep over epsilon (IQAE precision) and alpha (confidence level)
to analyze the resource requirements (shots and Grover calls) for VaR estimation.
"""

import csv
import os
from typing import List, Tuple

import numpy as np
import scipy.stats

from classiq import *
from classiq.applications.iqae.iqae import IQAE


# Portfolio parameters
MU = 0.15  # Expected return
SIGMA = 0.20  # Volatility
NUM_QUBITS = 7  # Discretization resolution

# Global threshold used by quantum payoff oracle
GLOBAL_INDEX: int = 0


def get_log_normal_probabilities(
    mu: float, sigma: float, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate log-normal probability distribution for asset values."""
    log_normal_mean = np.exp(mu + sigma**2 / 2)
    log_normal_variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    log_normal_stddev = np.sqrt(log_normal_variance)

    # Cut distribution at ±3σ from mean
    low = np.maximum(0, log_normal_mean - 3 * log_normal_stddev)
    high = log_normal_mean + 3 * log_normal_stddev
    
    x = np.linspace(low, high, num_points)
    probs = scipy.stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    
    return x, probs


def calculate_classical_var(
    grid_points: np.ndarray, probs: np.ndarray, confidence_level: float
) -> float:
    """Calculate VaR classically using cumulative distribution."""
    accumulated = 0.0
    for i, p in enumerate(probs):
        accumulated += p
        if accumulated > confidence_level:
            return grid_points[i]
    return grid_points[-1]


def get_var_index(probs: np.ndarray, confidence_level: float) -> int:
    """Get the index corresponding to VaR at given confidence level."""
    accumulated = 0.0
    for i, p in enumerate(probs):
        accumulated += p
        if accumulated > confidence_level:
            return i
    return len(probs) - 1


# Quantum circuit definitions
@qfunc
def load_distribution(asset: QNum, probs: List[float]):
    """Load probability distribution into quantum state."""
    inplace_prepare_state(probs, bound=0, target=asset)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    """Payoff function: mark states below threshold."""
    ind ^= asset < GLOBAL_INDEX


@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit, probs: List[float]):
    """Combined state preparation for IQAE."""
    load_distribution(asset=asset, probs=probs)
    payoff(asset=asset, ind=ind)


def create_iqae_instance(probs: List[float], num_qubits: int) -> IQAE:
    """Create IQAE instance with state preparation."""
    # Wrap state_preparation to bind probs
    @qfunc(synthesize_separately=True)
    def bound_state_prep(asset: QArray[QBit], ind: QBit):
        state_preparation(asset=asset, ind=ind, probs=probs)
    
    iqae = IQAE(
        state_prep_op=bound_state_prep,
        problem_vars_size=num_qubits,
        constraints=Constraints(max_width=28),
        preferences=Preferences(machine_precision=num_qubits),
    )
    return iqae


def run_iqae_estimation(
    iqae: IQAE, epsilon: float, alpha_iqae: float
) -> dict:
    """Run single IQAE estimation and extract metrics."""
    iqae_res = iqae.run(epsilon=epsilon, alpha=alpha_iqae)
    
    measured_payoff = iqae_res.estimation
    ci_low, ci_high = iqae_res.confidence_interval
    
    # Extract resource usage
    iterations_data = getattr(iqae_res, "iterations_data", []) or []
    shots_total = 0
    grover_calls = 0
    
    for iteration in iterations_data:
        k = getattr(iteration, "grover_iterations", None)
        shots = None
        
        if hasattr(iteration, "sample_results") and iteration.sample_results is not None:
            shots = getattr(iteration.sample_results, "num_shots", None)
        if hasattr(iteration, "num_shots") and shots is None:
            shots = iteration.num_shots
            
        if shots is not None:
            shots_total += shots
            if k is not None:
                grover_calls += k * shots
    
    # Fallback for shots
    if shots_total == 0 and hasattr(iqae_res, "sample_results"):
        shots_total = getattr(iqae_res.sample_results, "num_shots", 0)
    
    return {
        "estimation": measured_payoff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "shots_total": shots_total,
        "grover_calls": grover_calls,
    }


def var_parameter_sweep(
    epsilons: List[float],
    confidence_levels: List[float],
    mu: float = MU,
    sigma: float = SIGMA,
    num_qubits: int = NUM_QUBITS,
    alpha_iqae: float = 0.01,
    output_path: str = "results/var_sweep.csv",
):
    """
    Perform parameter sweep over epsilon and confidence levels.
    
    Args:
        epsilons: List of IQAE precision parameters
        confidence_levels: List of VaR confidence levels (e.g., [0.01, 0.05, 0.10])
        mu: Portfolio expected return
        sigma: Portfolio volatility
        num_qubits: Number of qubits for discretization
        alpha_iqae: IQAE confidence parameter (not VaR confidence)
        output_path: CSV output file path
    """
    global GLOBAL_INDEX
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate probability distribution
    grid_points, prob_density = get_log_normal_probabilities(mu, sigma, 2**num_qubits)
    probs = (prob_density / np.sum(prob_density)).tolist()
    
    # CSV structure
    fieldnames = [
        "epsilon",
        "confidence_level",
        "shots",
        "grover_calls",
        "VaR_theoretical",
        "VaR_predicted",
        "mu",
        "sigma",
        "alpha_iqae",
    ]
    
    rows = []
    
    print(f"Starting sweep: {len(epsilons)} epsilons × {len(confidence_levels)} confidence levels")
    
    for conf_level in confidence_levels:
        print(f"\n=== Confidence Level: {conf_level:.3f} ===")
        
        # Calculate classical VaR
        var_theoretical = calculate_classical_var(grid_points, probs, conf_level)
        var_index = get_var_index(probs, conf_level)
        GLOBAL_INDEX = int(var_index)
        
        print(f"Classical VaR: {var_theoretical:.4f} (index {var_index})")
        
        # Create IQAE instance for this confidence level
        iqae = create_iqae_instance(probs, num_qubits)
        
        for eps in epsilons:
            print(f"  epsilon={eps:.3f}...", end=" ")
            
            try:
                result = run_iqae_estimation(iqae, epsilon=eps, alpha_iqae=alpha_iqae)
                
                # Convert IQAE probability estimate to VaR value
                # IQAE estimates P(asset < threshold), find corresponding grid point
                estimated_prob = result["estimation"]
                var_predicted_index = get_var_index(probs, estimated_prob)
                var_predicted = grid_points[var_predicted_index]
                
                rows.append({
                    "epsilon": eps,
                    "confidence_level": conf_level,
                    "shots": result["shots_total"],
                    "grover_calls": result["grover_calls"],
                    "VaR_theoretical": var_theoretical,
                    "VaR_predicted": var_predicted,
                    "mu": mu,
                    "sigma": sigma,
                    "alpha_iqae": alpha_iqae,
                })
                
                print(f"shots={result['shots_total']}, grover_calls={result['grover_calls']}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({
                    "epsilon": eps,
                    "confidence_level": conf_level,
                    "shots": None,
                    "grover_calls": None,
                    "VaR_theoretical": var_theoretical,
                    "VaR_predicted": None,
                    "mu": mu,
                    "sigma": sigma,
                    "alpha_iqae": alpha_iqae,
                })
    
    # Write results
    # if file exists, don't overwrite, make a new file
    if os.path.exists(output_path):
        base, ext = os.path.splitext(output_path)
        # maybe_postfi
        base_parts = base.split("_")
        if base_parts[-1].isdigit():
            base_parts[-1] = f"{int(base_parts[-1]) + 1:02d}"
        else:
            base_parts.append("01")
        new_base = "_".join(base_parts)
        output_path = f"{new_base}{ext}"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    # Define sweep parameters
    epsilons = [0.1, 0.075, 0.05, 0.025, 0.0125, 0.01, 0.001]
    var_alphas = [0.01, 0.2, 0.3, 0.4, 0.5]
    
    var_parameter_sweep(
        epsilons=epsilons,
        confidence_levels=var_alphas,
        mu=0.15,
        sigma=0.20,
        num_qubits=7,
        alpha_iqae=0.01,  # IQAE's internal confidence (separate from VaR confidence)
        output_path="results/var_sweep.csv",
    )