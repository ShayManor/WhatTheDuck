"""
MONTE CARLO VaR ANALYSIS - PUBLICATION-QUALITY VISUALIZATION
============================================================
Advanced Monte Carlo simulation supporting:
- Multi-day horizons
- Non-Gaussian distributions (Normal, Student-t, Skew-Normal)
- Temporal correlation (AR(1) process)
- Comprehensive convergence analysis

Author: Classical Monte Carlo Analysis
Date: 2026
"""

import os
import numpy as np
from scipy.stats import skewnorm
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import alive_progress


# Market parameters
mu = 0.15                      # Mean daily return (15%)
sigma = 0.20                   # Daily volatility (20%)
confidence_level = 0.95        # VaR confidence level

# Multi-day and distribution settings
T = 5                          # Number of days for multi-day VaR
dist = "skewnorm"              # Distribution: "gaussian", "student-t", "skewnorm"
df = 3                         # Degrees of freedom for Student-t
skew_alpha = 7.0               # Skew parameter for skew-normal
rho = 0.6                      # AR(1) correlation coefficient

# Where modeling error is intentionally held fixed
# You do not mix models during convergence plots.

# Simulation settings
num_samples_max = 10**7        # Maximum samples
num_samples_count = 50        # Number of sample sizes to test
theoretical_N = num_samples_max * 2 # Samples for theoretical VaR estimation
theoretical_estimations = 1        # Averaging runs for theoretical VaR
CPU_WORKERS = 10               # Parallel workers

OUTPUT = '../graphs/data/monte_carlo_naive2.csv'

# Generate logarithmically spaced sample sizes
num_samples_list = np.unique(
    np.logspace(1, np.log10(num_samples_max), num_samples_count, dtype=int)
).tolist()

# Make directories for output
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ============================================================================
# MONTE CARLO ENGINE
# ============================================================================

@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results."""
    num_samples: int
    var_estimate: float
    error: float


def monte_carlo_var(
    N: int,
    mu: float,
    sigma: float,
    confidence_level: float,
    theoretical_var: float,
    T: int = 1,
    dist: str = "gaussian",
    df: int = 5,
    skew_alpha: float = 0.0,
    rho: float = 0.0
) -> MonteCarloResult:
    """
    Monte Carlo VaR estimation with advanced features.
    
    Parameters:
        N: Number of Monte Carlo scenarios
        mu: Mean daily return
        sigma: Daily volatility
        confidence_level: Confidence level for VaR
        theoretical_var: Theoretical VaR (for error calculation)
        T: Number of days in horizon
        dist: Distribution type ("gaussian", "student-t", "skewnorm")
        df: Degrees of freedom for Student-t
        skew_alpha: Skew parameter for skew-normal
        rho: AR(1) correlation coefficient
    
    Returns:
        MonteCarloResult with num_samples, var_estimate, and error
    """
    # Generate daily returns based on distribution
    if dist == "gaussian":
        daily_returns = np.random.normal(mu, sigma, size=(N, T))
    elif dist == "student-t":
        daily_returns = mu + sigma * np.random.standard_t(df, size=(N, T))
    elif dist == "skewnorm":
        daily_returns = mu + sigma * skewnorm.rvs(skew_alpha, size=(N, T))
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    
    # Apply AR(1) correlation if specified
    if rho != 0.0 and T > 1:
        correlated_returns = np.zeros_like(daily_returns)
        correlated_returns[:, 0] = daily_returns[:, 0]
        for t in range(1, T):
            innovation = (daily_returns[:, t] - mu) * np.sqrt(1 - rho**2)
            correlated_returns[:, t] = mu + rho * (correlated_returns[:, t-1] - mu) + innovation
        daily_returns = correlated_returns
    
    # Aggregate multi-day returns and compute VaR
    total_returns = daily_returns.sum(axis=1)
    # losses = -total_returns
    var_estimate = np.quantile(total_returns, 1 - confidence_level)
    error = abs(var_estimate - theoretical_var)
    
    return MonteCarloResult(N, var_estimate, error)


# Estimate theoretical VaR using large N and multiple runs
print(f"\nEstimating theoretical VaR with N={theoretical_N:,} over {theoretical_estimations} runs...")
theoretical_vars = []
for i in range(theoretical_estimations):
    mc_run = monte_carlo_var(
        theoretical_N, mu, sigma, confidence_level, 0.0, T, dist, df, skew_alpha, rho
    )
    theoretical_vars.append(mc_run.var_estimate)
    print(f"  Run {i+1}/{theoretical_estimations}: VaR = {mc_run.var_estimate:.5f}")
theoretical_var = float(np.mean(theoretical_vars))
print(f"Theoretical VaR ({int(confidence_level*100)}%): {theoretical_var:.5f}")
# This does two critical things:
# It removes modeling error from the convergence study
# It isolates probability estimation error only

# Run parallel simulations
print(f"\nRunning {len(num_samples_list)} simulations in parallel...")
mc_results = []
total_completed = 0

with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
    futures = [
        executor.submit(
            monte_carlo_var, N, mu, sigma, confidence_level, theoretical_var,
            T, dist, df, skew_alpha, rho
        )
        for N in num_samples_list
    ]
    
    for future in as_completed(futures):
        result = future.result()
        mc_results.append(result)
        total_completed += 1
        percent_complete = (total_completed / len(num_samples_list)) * 100
        print(f"  {percent_complete:5.1f}% | N={result.num_samples:>9,} | "
              f"VaR={result.var_estimate:.5f} | Error={result.error:.3e}")

# Sort results and extract data
mc_results.sort(key=lambda x: x.num_samples)
# num_samples_list, var_results, errors = zip(*[
#     (r.num_samples, r.var_estimate, r.error) for r in mc_results
# ])

# ============================================================================
# WRITE CSV RESULTS
# ============================================================================

CSV_HEADERS = [
    "Epsilon",
    "N",
    "VaR_prediction",
    "VaR_theoretical",
    "mu", "sigma", "confidence_level", "T", "dist", "df", "skew_alpha", "rho"
]

print(f"\nWriting results to {OUTPUT}...")

# Episilon = error
with alive_progress.alive_bar(len(mc_results)) as bar:
    with open(OUTPUT, 'w') as f:
        f.write(','.join(CSV_HEADERS) + '\n')
        for result in mc_results:
            row = [
                f"{result.error:.6e}",
                str(result.num_samples),
                f"{result.var_estimate:.6f}",
                f"{theoretical_var:.6f}",
                f"{mu:.6f}",
                f"{sigma:.6f}",
                f"{confidence_level:.6f}",
                str(T),
                dist,
                str(df),
                f"{skew_alpha:.6f}",
                f"{rho:.6f}"
            ]
            f.write(','.join(row) + '\n')
            bar()