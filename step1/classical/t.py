import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.font_manager as fm

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Professional color palette - deep academic blues with warm accent
COLOR_PRIMARY = '#1e3a8a'      # Deep blue
COLOR_SECONDARY = '#3b82f6'    # Bright blue
COLOR_ACCENT = '#f59e0b'       # Amber accent
COLOR_DANGER = '#dc2626'       # Red for theoretical line
COLOR_GRID = '#e5e7eb'         # Light gray grid
COLOR_TEXT = '#1f2937'         # Dark gray text
COLOR_BOUND_UPPER = '#6366f1'  # Indigo for upper bound
COLOR_BOUND_LOWER = '#8b5cf6'  # Purple for lower bound

# Parameters ------------------------------------------------------------------#
mu = 0.15      # mean daily return (15%)
sigma = 0.20   # daily volatility (20%)
confidence_level = 0.95
num_samples_max = 10**8
num_samples_count = 200
num_samples_list = np.unique(
    np.logspace(1, np.log10(num_samples_max), num_samples_count, dtype=int)
).tolist()
CPU_WORKERS = 10


# Make output dir if not exists -----------------------------------------------#
output_dir = './outputs'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Simulate returns ------------------------------------------------------------#
@dataclass
class MonteCarloResult:
    num_samples: int
    var_estimate: float
    error: float

def monte_carlo_var(
    N: int,
    mu: float,
    sigma: float,
    confidence_level: float,
    theoretical_var: float
) -> MonteCarloResult:
    """Run one Monte Carlo VaR estimation for a given sample size N."""
    returns = np.random.normal(mu, sigma, N)
    losses = -returns
    var_estimate = np.quantile(losses, confidence_level)
    error = abs(var_estimate - theoretical_var)
    return MonteCarloResult(N, var_estimate, error)

# Run monte carlo -------------------------------------------------------------#
mc_results = []
theoretical_var = - (mu + sigma * norm.ppf(1 - confidence_level))
print(f"Theoretical VaR({int(confidence_level*100)}%) = {theoretical_var:.5f}")

total_completed = 0
with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
    futures = [executor.submit(monte_carlo_var, N, mu, sigma, confidence_level, theoretical_var)
               for N in num_samples_list]
    
    for future in as_completed(futures):
        result = future.result()
        mc_results.append(result)
        
        total_completed += 1
        percent_complete = (total_completed / len(num_samples_list)) * 100
        print(f"{percent_complete:.2f}% - Samples: {result.num_samples:>8}, VaR ≈ {result.var_estimate:.5f}, Error ≈ {result.error:.5f}")

mc_results.sort(key=lambda x: x.num_samples)
num_samples_list, var_results, errors = zip(*[(r.num_samples, r.var_estimate, r.error) for r in mc_results])

# Calculate reference lines
scaled_errors = np.array(errors) * np.sqrt(np.array(num_samples_list))
witness_n = np.median(scaled_errors)
ref_line_n = witness_n / np.sqrt(num_samples_list)
# witness_top_n = np.percentile(scaled_errors, 95) 
# witness_bottom_n = np.percentile(scaled_errors, 5)
# ref_line_top_n = witness_top_n / np.sqrt(num_samples_list)
# ref_line_bottom_n = witness_bottom_n / np.sqrt(num_samples_list)

inv_error_sq = 1 / np.array(errors)**2
num_samples_array = np.array(num_samples_list)
scaled_samples = num_samples_array / inv_error_sq
witness_e2 = np.median(scaled_samples)
# witness_top_e2 = np.percentile(scaled_samples, 95)
# witness_bottom_e2 = np.percentile(scaled_samples, 5)
# ref_line_top_e2 = witness_top_e2 * inv_error_sq
# ref_line_bottom_e2 = witness_bottom_e2 * inv_error_sq
ref_line_e2 = witness_e2 * inv_error_sq

# Create professional visualizations ------------------------------------------#

# Figure 1: VaR Convergence
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('#fafafa')

# Plot data with enhanced styling
ax1.plot(num_samples_list, var_results, 
         marker='o', markersize=4, linewidth=2, 
         color=COLOR_PRIMARY, alpha=0.8,
         label='Monte Carlo Estimate', zorder=3)
ax1.axhline(y=theoretical_var, color=COLOR_DANGER, 
            linestyle='--', linewidth=2.5, alpha=0.9,
            label='Theoretical VaR', zorder=2)

# Enhanced grid
ax1.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID, zorder=1)
ax1.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID, zorder=1)

# Styling
ax1.set_xscale('log')
ax1.set_xlabel('Number of Samples (N)', fontweight='500', color=COLOR_TEXT)
ax1.set_ylabel(f'Value-at-Risk ({int(confidence_level*100)}% Confidence)', 
               fontweight='500', color=COLOR_TEXT)
ax1.set_title('Convergence of Classical Monte Carlo VaR Estimation', 
              fontweight='600', color=COLOR_TEXT, pad=20)

# Enhanced legend
legend1 = ax1.legend(loc='best', frameon=True, fancybox=True, 
                     shadow=True, framealpha=0.95, edgecolor=COLOR_GRID)
legend1.get_frame().set_facecolor('white')

# Spines styling
for spine in ax1.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

ax1.tick_params(colors=COLOR_TEXT)

plt.tight_layout()
plt.savefig('./outputs/01_var_convergence.png', 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 01_var_convergence.png")

# Figure 2: Error Scaling
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('white')
ax2.set_facecolor('#fafafa')

# Plot main data
ax2.plot(num_samples_list, errors, 
         marker='o', markersize=4, linewidth=2, 
         color=COLOR_SECONDARY, alpha=0.8,
         label='Absolute Error', zorder=4)

# Reference lines with improved styling
ax2.plot(num_samples_list, ref_line_n,
         color=COLOR_BOUND_UPPER, linestyle='--', linewidth=2, alpha=0.7,
         label=r'$\mathcal{O}(N^{-1/2})$', zorder=3)
# ax2.plot(num_samples_list, ref_line_bottom_n, 
#          color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=2, alpha=0.7,
#          label=r'$\Omega(N^{-1/2})$ Lower Bound', zorder=3)

# Fill between bounds for visual clarity
# ax2.fill_between(num_samples_list, ref_line_bottom_n, ref_line_top_n,
#                   alpha=0.1, color=COLOR_PRIMARY, zorder=1)

# Enhanced grid
ax2.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID, zorder=2)
ax2.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID, zorder=2)

# Styling
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Number of Samples (N)', fontweight='500', color=COLOR_TEXT)
ax2.set_ylabel('Absolute Error |VaR - Theoretical|', fontweight='500', color=COLOR_TEXT)
ax2.set_title('Monte Carlo Error Scaling: Classical Convergence Rate', 
              fontweight='600', color=COLOR_TEXT, pad=20)

# Enhanced legend
legend2 = ax2.legend(loc='best', frameon=True, fancybox=True, 
                     shadow=True, framealpha=0.95, edgecolor=COLOR_GRID)
legend2.get_frame().set_facecolor('white')

# Spines styling
for spine in ax2.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

ax2.tick_params(colors=COLOR_TEXT)

# Add annotation for convergence rate
mid_idx = len(num_samples_list) // 2
ax2.annotate(r'Error $\propto N^{-1/2}$', 
             xy=(num_samples_list[mid_idx], errors[mid_idx]),
             xytext=(num_samples_list[mid_idx] * 0.1, errors[mid_idx] * 3),
             fontsize=11, color=COLOR_ACCENT, fontweight='600',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=COLOR_ACCENT, alpha=0.9),
             arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

plt.tight_layout()
plt.savefig('./outputs/02_error_scaling.png', 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 02_error_scaling.png")

# Figure 3: Sample Complexity
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor('white')
ax3.set_facecolor('#fafafa')

# Plot main relationship
ax3.plot(inv_error_sq, num_samples_list, 
         marker='o', markersize=4, linewidth=2, 
         color=COLOR_PRIMARY, alpha=0.8,
         label='Observed Samples vs $\\varepsilon^{-2}$', zorder=4)

# Reference lines
ax3.plot(inv_error_sq, ref_line_e2, 
         color=COLOR_BOUND_UPPER, linestyle='--', linewidth=2, alpha=0.7,
         label=r'$\mathcal{O}(\varepsilon^{-2})$', zorder=3)
# ax3.plot(inv_error_sq, ref_line_bottom_e2, 
#          color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=2, alpha=0.7,
#          label=r'$\Omega(\varepsilon^{-2})$ Lower Bound', zorder=3)

# Fill between bounds
# ax3.fill_between(inv_error_sq, ref_line_bottom_e2, ref_line_top_e2,
#                   alpha=0.1, color=COLOR_PRIMARY, zorder=1)

# Enhanced grid
ax3.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID, zorder=2)
ax3.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID, zorder=2)

# Styling
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel(r'Inverse Squared Error ($\varepsilon^{-2}$)', 
               fontweight='500', color=COLOR_TEXT)
ax3.set_ylabel('Required Number of Samples (N)', fontweight='500', color=COLOR_TEXT)
ax3.set_title('Sample Complexity: Classical Monte Carlo Scaling', 
              fontweight='600', color=COLOR_TEXT, pad=20)

# Enhanced legend
legend3 = ax3.legend(loc='best', frameon=True, fancybox=True, 
                     shadow=True, framealpha=0.95, edgecolor=COLOR_GRID)
legend3.get_frame().set_facecolor('white')

# Spines styling
for spine in ax3.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

ax3.tick_params(colors=COLOR_TEXT)

# Add annotation for complexity
mid_idx = len(inv_error_sq) // 2
ax3.annotate(r'$N \propto \varepsilon^{-2}$', 
             xy=(inv_error_sq[mid_idx], num_samples_list[mid_idx]),
             xytext=(inv_error_sq[mid_idx] * 0.15, num_samples_list[mid_idx] * 5),
             fontsize=11, color=COLOR_ACCENT, fontweight='600',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=COLOR_ACCENT, alpha=0.9),
             arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

plt.tight_layout()
plt.savefig('./outputs/03_sample_complexity.png', 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 03_sample_complexity.png")

# Create a combined summary figure
fig4 = plt.figure(figsize=(16, 10))
fig4.patch.set_facecolor('white')
gs = fig4.add_gridspec(2, 2, hspace=0.3, wspace=0.3, 
                       top=0.93, bottom=0.07, left=0.08, right=0.95)

# Add overall title
fig4.suptitle('Classical Monte Carlo Analysis: VaR Estimation & Convergence Properties', 
              fontsize=18, fontweight='700', color=COLOR_TEXT, y=0.98)

# Subplot 1: VaR Convergence
ax4_1 = fig4.add_subplot(gs[0, :])
ax4_1.set_facecolor('#fafafa')
ax4_1.plot(num_samples_list, var_results, marker='o', markersize=3, 
           linewidth=2, color=COLOR_PRIMARY, alpha=0.8, label='Monte Carlo Estimate')
ax4_1.axhline(y=theoretical_var, color=COLOR_DANGER, linestyle='--', 
              linewidth=2.5, alpha=0.9, label='Theoretical VaR')
ax4_1.set_xscale('log')
ax4_1.set_xlabel('Number of Samples (N)', fontweight='500', color=COLOR_TEXT)
ax4_1.set_ylabel(f'VaR ({int(confidence_level*100)}%)', fontweight='500', color=COLOR_TEXT)
ax4_1.set_title('(A) VaR Estimate Convergence', fontweight='600', 
                color=COLOR_TEXT, loc='left')
ax4_1.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax4_1.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
legend = ax4_1.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95)
legend.get_frame().set_facecolor('white')
for spine in ax4_1.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Subplot 2: Error Scaling
ax4_2 = fig4.add_subplot(gs[1, 0])
ax4_2.set_facecolor('#fafafa')
ax4_2.plot(num_samples_list, errors, marker='o', markersize=3, 
           linewidth=2, color=COLOR_SECONDARY, alpha=0.8, label='Absolute Error')
ax4_2.plot(num_samples_list, ref_line_top_n, color=COLOR_BOUND_UPPER, 
           linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\mathcal{O}(N^{-1/2})$')
ax4_2.plot(num_samples_list, ref_line_bottom_n, color=COLOR_BOUND_LOWER, 
           linestyle='-.', linewidth=1.5, alpha=0.7, label=r'$\Omega(N^{-1/2})$')
ax4_2.fill_between(num_samples_list, ref_line_bottom_n, ref_line_top_n,
                    alpha=0.1, color=COLOR_PRIMARY)
ax4_2.set_xscale('log')
ax4_2.set_yscale('log')
ax4_2.set_xlabel('Samples (N)', fontweight='500', color=COLOR_TEXT)
ax4_2.set_ylabel('Absolute Error', fontweight='500', color=COLOR_TEXT)
ax4_2.set_title('(B) Error Scaling', fontweight='600', color=COLOR_TEXT, loc='left')
ax4_2.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax4_2.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
legend = ax4_2.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95, fontsize=9)
legend.get_frame().set_facecolor('white')
for spine in ax4_2.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Subplot 3: Sample Complexity
ax4_3 = fig4.add_subplot(gs[1, 1])
ax4_3.set_facecolor('#fafafa')
ax4_3.plot(inv_error_sq, num_samples_list, marker='o', markersize=3, 
           linewidth=2, color=COLOR_PRIMARY, alpha=0.8, label='Observed')
ax4_3.plot(inv_error_sq, ref_line_top_e2, color=COLOR_BOUND_UPPER, 
           linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\mathcal{O}(\varepsilon^{-2})$')
ax4_3.plot(inv_error_sq, ref_line_bottom_e2, color=COLOR_BOUND_LOWER, 
           linestyle='-.', linewidth=1.5, alpha=0.7, label=r'$\Omega(\varepsilon^{-2})$')
ax4_3.fill_between(inv_error_sq, ref_line_bottom_e2, ref_line_top_e2,
                    alpha=0.1, color=COLOR_PRIMARY)
ax4_3.set_xscale('log')
ax4_3.set_yscale('log')
ax4_3.set_xlabel(r'$\varepsilon^{-2}$', fontweight='500', color=COLOR_TEXT)
ax4_3.set_ylabel('Samples (N)', fontweight='500', color=COLOR_TEXT)
ax4_3.set_title('(C) Sample Complexity', fontweight='600', color=COLOR_TEXT, loc='left')
ax4_3.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax4_3.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
legend = ax4_3.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95, fontsize=9)
legend.get_frame().set_facecolor('white')
for spine in ax4_3.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

plt.savefig('./outputs/04_combined_analysis.png', 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 04_combined_analysis.png")

print("\n" + "="*60)
print("All publication-quality graphs generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  • 01_var_convergence.png     - VaR estimate convergence")
print("  • 02_error_scaling.png       - Error scaling analysis")
print("  • 03_sample_complexity.png   - Sample complexity visualization")
print("  • 04_combined_analysis.png   - Comprehensive summary figure")
print("\nThese graphs are ready for your presentation!")

plt.show()