import numpy as np
import matplotlib.pyplot as plt

# Parameters ------------------------------------------------------------------#
# np.random.seed(42)  # reproducibility

mu = 0.7      # mean daily return
sigma = 0.13  # daily volatility
confidence_level = 0.95
num_samples_list = [10**i for i in range(1, 8)]


# simulate_returns ------------------------------------------------------------#			
def simulate_returns(mu, sigma, num_samples):
    """
    Simulate daily returns from a Gaussian distribution.
    """
    return np.random.normal(mu, sigma, num_samples)


# compute_var -----------------------------------------------------------------#			
def compute_var(returns, confidence_level):
    """
    Compute Value at Risk (VaR) at the given confidence level.
    VaR is the negative of the quantile of returns.
    """
    var = -np.quantile(returns, 1 - confidence_level)
    return var

# Run monte carlo -------------------------------------------------------------#
results = []

for N in num_samples_list:
    returns = simulate_returns(mu, sigma, N)
    var_estimate = compute_var(returns, confidence_level)
    results.append(var_estimate)
    print(f"Samples: {N:>7}, VaR({int(confidence_level*100)}%) ≈ {var_estimate:.5f}")


# Plot ------------------------------------------------------------------------#
# For Gaussian, VaR at confidence level alpha:
from scipy.stats import norm
theoretical_var = - (mu + sigma * norm.ppf(1 - confidence_level))
print(f"Theoretical VaR({int(confidence_level*100)}%) = {theoretical_var:.5f}")

plt.figure(figsize=(8,5))
plt.plot(num_samples_list, results, marker='o', label='Monte Carlo VaR estimate')
plt.xscale('log')
plt.xlabel("Number of Samples (log scale)")
plt.ylabel(f"VaR at {int(confidence_level*100)}% confidence")

# Show theoretical VaR line
plt.axhline(y=theoretical_var, color='r', linestyle='--', label='Theoretical VaR')

plt.title("Convergence of Classical Monte Carlo VaR")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()


