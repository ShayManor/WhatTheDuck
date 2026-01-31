import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parameters ------------------------------------------------------------------#
# np.random.seed(42)  # reproducibility

mu = 0.15      # mean daily return (0.1%)
sigma = 0.20   # daily volatility (2%)
confidence_level = 0.95
# num_samples_list = [n for n in range(10, 10**8 + 1, 100)]
num_samples_max = 10**8
num_samples_count = 200
num_samples_list = np.unique(
    np.logspace(1, np.log10(num_samples_max), num_samples_count, dtype=int)
).tolist()  # 10, ..., 10^8

CPU_WORKERS = 10

# Simulate returns ------------------------------------------------------------#
@dataclass
class MonteCarloResult:
    num_samples: int
    var_estimate: float
    error: float

def monte_carlo_var(
    N:int,
    mu: float,
    sigma: float,
    confidence_level: float,
    theoretical_var: float
) -> MonteCarloResult:
    """
    Run one Monte Carlo VaR estimation for a given sample size N.
    Returns: (N, var_estimate, error)
    """
    returns = np.random.normal(mu, sigma, N)
    losses = -returns
    var_estimate = np.quantile(losses, confidence_level)
    error = abs(var_estimate - theoretical_var)
    return MonteCarloResult(N, var_estimate, error)
    

# Run monte carlo - c ---------------------------------------------------------#
mc_results = []

# Theoretical VaR for Gaussian
theoretical_var = - (mu + sigma * norm.ppf(1 - confidence_level))
print(f"Theoretical VaR({int(confidence_level*100)}%) = {theoretical_var:.5f}")

total_completed = 0


with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
    # Submit all tasks in parallel
    futures = [executor.submit(monte_carlo_var, N, mu, sigma, confidence_level, theoretical_var)
               for N in num_samples_list]

    # Collect results as they complete
    for future in as_completed(futures):
        result = future.result()
        mc_results.append(result)
        
        total_completed += 1
        percent_complete = (total_completed / len(num_samples_list)) * 100
        
        print(f"{percent_complete:.2f}% - Samples: {result.num_samples:>8}, VaR ≈ {result.var_estimate:.5f}, Error ≈ {result.error:.5f}")
        
mc_results.sort(key=lambda x: x.num_samples)
num_samples_list, var_results, errors = zip(*[(r.num_samples, r.var_estimate, r.error) for r in mc_results])

# O(1/sqrt(N)) relation
witness_top = np.max(errors * np.sqrt(num_samples_list))    # ensures all errors <= witness / sqrt(N)
witness_bottom = np.min(errors * np.sqrt(num_samples_list)) # ensures all errors >= witness / sqrt(N)

ref_line_top = witness_top / np.sqrt(num_samples_list)
ref_line_bottom = witness_bottom / np.sqrt(num_samples_list)

# Plot convergence and demonstrate O(1/ε²) scaling - d ------------------------#

# Plot 1: Convergence of VaR estimate
plt.figure(figsize=(6,4))
plt.plot(num_samples_list, var_results, marker='o', label='Monte Carlo VaR estimate')
plt.axhline(y=theoretical_var, color='r', linestyle='--', label='Theoretical VaR')
plt.xscale('log')
plt.xlabel("Number of Samples (log scale)")
plt.ylabel(f"VaR at {int(confidence_level*100)}% confidence")
plt.title("Convergence of Classical Monte Carlo VaR")
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()

# Plot 2: Error scaling vs samples
plt.figure(figsize=(6,4))
plt.plot(num_samples_list, errors, marker='o', label='|VaR estimate - Theory|')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Absolute Error")
plt.title("Monte Carlo Error Scaling")

# Reference lines for O(1/sqrt(N))
plt.plot(num_samples_list, ref_line_top, 'k--', label=r'O(1/$\sqrt{{N}}$) upper bound')
plt.plot(num_samples_list, ref_line_bottom, 'k-.', label=r'Ω(1/$\sqrt{{N}}$) lower bound')

# Plot 3: O(1/E^2) scaling
inv_error_sq = 1 / np.array(errors)**2

plt.figure(figsize=(6,4))
plt.plot(inv_error_sq, num_samples_list, marker='o', label='Observed N vs 1/ε²')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$1/\varepsilon^2$')
plt.ylabel("Number of Samples N")
plt.title("Monte Carlo Sample Requirement vs 1/ε²")
plt.grid(True, which='both', ls='--', alpha=0.5)

# Reference line: 1/ε²
slope = num_samples_list[-1] / inv_error_sq[-1]
plt.plot(inv_error_sq, slope*inv_error_sq, 'r--', label='Reference O(1/ε²)')

plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()

plt.show()
