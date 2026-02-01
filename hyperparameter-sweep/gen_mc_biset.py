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
import alive_progress

from classical_monte_carlo import estimate_var_classical
from sweep import get_distribution

def get_val(epsilon, alpha=0.05):
    import scipy.stats as st
    grid, probs = get_distribution("normal", mu=1.5, sigma=0.2, num_points=512)
    true_var = st.norm.ppf(0.05, loc=1.5, scale=0.2)
    result = estimate_var_classical(
        probs,
        grid,
        alpha=alpha,
        seed=np.random.randint(1, 10000),
        epsilon=epsilon
    )

    return result['var'], true_var, result['total_samples']



# Market parameters
# Market parameters
mu = 0.15                      # Mean daily return (15%)
sigma = 0.20                   # Daily volatility (20%)
confidence_level = 0.95        # VaR confidence level

# Multi-day and distribution settings
T = 1                          # Number of days for multi-day VaR
dist = "skewnorm"              # Distribution: "gaussian", "student-t", "skewnorm"
df = 3                         # Degrees of freedom for Student-t
skew_alpha = 7.0               # Skew parameter for skew-normal
rho = 0.0                      # AR(1) correlation coefficient

OUTPUT = '../graphs/data/monte_carlo_bisect.csv'

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

results = []    
E_MAX = 0.1
E_MIN = 0.001
epsilon_values = np.logspace(np.log10(E_MAX), np.log10(E_MIN), num=10)
mc_results = []
with alive_progress.alive_bar(len(epsilon_values)) as bar:
	for epsilon in epsilon_values:
		var_estimate, theoretical_var, num_samples = get_val(epsilon, alpha=1 - confidence_level)
		print(f"Epsilon: {epsilon:.6e}, Samples: {num_samples}, VaR Estimate: {var_estimate:.6f}, Theoretical VaR: {theoretical_var:.6f}")
		mc_results.append(
			MonteCarloResult(
				error=epsilon,
				num_samples=num_samples,
				var_estimate=var_estimate
			)
		)
		bar()


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