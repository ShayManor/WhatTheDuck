# Context: Classical Monte Carlo VaR Estimation

## Purpose
This script demonstrates classical Monte Carlo estimation of **Value-at-Risk (VaR)** for a single asset over a fixed time horizon (e.g., 1 day). It illustrates how Monte Carlo estimates converge to the theoretical VaR as a function of the number of samples and validates the expected **O(1/√N)** error scaling and **O(ε⁻²)** sample complexity.

---

## Methodology

1. **Asset Return Model**
   - Asset returns are modeled as a **Gaussian (normal) distribution** with parameters:
     - Mean daily return: `μ = 0.15` (15%)
     - Daily volatility: `σ = 0.20` (20%)

2. **Value-at-Risk (VaR)**
   - VaR at confidence level `α` (e.g., 95%) is defined as the **quantile of the loss distribution**:
     \[
     \text{VaR}_\alpha = - (\mu + \sigma \Phi^{-1}(1-\alpha))
     \]
   - The script computes both the **theoretical VaR** and **Monte Carlo estimates**.

3. **Monte Carlo Simulation**
   - For each sample size \(N\), simulate `N` daily returns from the Gaussian distribution.
   - Convert returns to losses and calculate the **quantile corresponding to the chosen confidence level**.
   - Compute the **absolute error** compared to the theoretical VaR.

4. **Parallel Computation**
   - Uses `ProcessPoolExecutor` with `CPU_WORKERS = 10` to run multiple sample sizes in parallel for efficiency.
   - Sample sizes span logarithmically from \(10^1\) to \(10^8\), with `num_samples_count = 250` unique values.

5. **Error and Sample Complexity Analysis**
   - **Error Scaling:** Checks that Monte Carlo absolute errors follow expected \(O(1/\sqrt{N})\) scaling.
   - **Sample Complexity:** Validates that the required number of samples \(N\) scales as \(O(\varepsilon^{-2})\) for a target error \(\varepsilon\).

---

## Outputs

The script generates **four main figures** in the `./outputs` directory:

1. **`01_var_convergence.png`**  
   - Convergence of Monte Carlo VaR estimates to the theoretical value as the number of samples increases.

2. **`02_error_scaling.png`**  
   - Absolute error versus sample size.
   - Reference lines showing theoretical \(O(N^{-1/2})\) convergence bounds.

3. **`03_sample_complexity.png`**  
   - Number of samples required as a function of inverse squared error (\(\varepsilon^{-2}\)).
   - Shows expected scaling of sample complexity.

4. **`04_combined_analysis.png`**  
   - Comprehensive figure combining:
     - VaR convergence
     - Error scaling
     - Sample complexity  

---

## Key Parameters

| Parameter                | Value                       | Description |
|---------------------------|----------------------------|-------------|
| `mu`                      | 0.15                        | Mean daily return |
| `sigma`                   | 0.20                        | Daily volatility |
| `confidence_level`        | 0.95                        | VaR confidence level |
| `num_samples_max`         | 1e8                         | Maximum Monte Carlo sample size |
| `num_samples_count`       | 250                         | Number of logarithmically spaced sample sizes |
| `CPU_WORKERS`             | 10                          | Number of parallel processes |

---

## Dependencies

- Python 3.8+  
- `numpy`  
- `scipy`  
- `matplotlib`  
- `dataclasses` (Python 3.7+)  
- `concurrent.futures`  

---

## Notes

- Uses professional plotting style with a **consistent color palette** and publication-quality figures.
- Outliers in errors are filtered for clearer visualizations in scaling plots.
- Demonstrates classical Monte Carlo behavior with clear statistical reference lines.

