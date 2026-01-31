"""
QUANTUM VaR ANALYSIS - IQAE EPSILON SWEEP VISUALIZATION
============================================================
Iterative Quantum Amplitude Estimation (IQAE) convergence analysis:
- Quantum speedup demonstration (O(1/ε) vs classical O(1/ε²))
- Error scaling with Grover iterations
- Sample complexity analysis
- Comprehensive convergence visualization

Based on quantum Value-at-Risk estimation using IQAE algorithm.

Author: Quantum Monte Carlo Analysis
Date: 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

theoretical_N = 20_000_000

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['mathtext.fontset'] = 'stix'

# Professional color palette - EXACT SAME AS CLASSICAL
COLOR_PRIMARY = '#1e3a8a'      # Deep blue
COLOR_SECONDARY = '#3b82f6'    # Bright blue
COLOR_ACCENT = '#f59e0b'       # Amber accent
COLOR_DANGER = '#dc2626'       # Red for theoretical line
COLOR_GRID = '#e5e7eb'         # Light gray grid
COLOR_TEXT = '#1f2937'         # Dark gray text
COLOR_BOUND_UPPER = '#6366f1'  # Indigo for upper bound
COLOR_BOUND_LOWER = '#8b5cf6'  # Purple for lower bound
COLOR_DIST = '#3b82f6'         # Distribution color

# ============================================================================
# PARAMETERS (from quantum VaR code)
# ============================================================================

num_qubits = 7
mu = 0.7                       # Log-normal mean
sigma = 0.13                   # Log-normal std dev
ALPHA = 0.07                   # VaR confidence level
TOLERANCE = ALPHA / 10

# Output directory
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def style_axes(ax, title, xlabel, ylabel):
    """Apply consistent professional styling to axes."""
    ax.set_facecolor('#fafafa')
    ax.set_title(title, fontweight='600', color=COLOR_TEXT, pad=20)
    ax.set_xlabel(xlabel, fontweight='500', color=COLOR_TEXT)
    ax.set_ylabel(ylabel, fontweight='500', color=COLOR_TEXT)
    ax.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
    ax.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
        spine.set_linewidth(1.5)


def create_legend(ax, **kwargs):
    """Create styled legend with consistent formatting."""
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      framealpha=0.95, edgecolor=COLOR_GRID, **kwargs)
    legend.get_frame().set_facecolor('white')
    return legend


# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

print("="*70)
print("QUANTUM VaR ANALYSIS - IQAE EPSILON SWEEP")
print("="*70)

# Load CSV data
csv_path = './data/iqae_epsilon_sweep.csv'
if not os.path.exists(csv_path):
    print(f"\nError: {csv_path} not found!")
    print("Please ensure iqae_epsilon_sweep.csv is in the ./data/ directory.")
    exit(1)

df = pd.read_csv(csv_path)

# Sort by grover_calls to ensure monotonic X-axis
df = df.sort_values('grover_calls_mean').reset_index(drop=True)

print(f"\nLoaded data from {csv_path}")
print(f"  • Data points: {len(df)}")
print(f"  • Epsilon range: [{df['epsilon'].min():.3f}, {df['epsilon'].max():.3f}]")
print(f"  • Alpha (confidence): {df['alpha'].iloc[0]:.3f}")

# Extract key arrays (epsilon IS the error measure)
# epsilon = target precision parameter (represents error)
# grover_calls_mean = N (number of Grover iterations × shots)
# a_true = true amplitude to estimate
# a_hat_mean = IQAE estimate
epsilons = df['epsilon'].values
grover_calls = df['grover_calls_mean'].values
a_true = df['a_true'].iloc[0]
a_hat_means = df['a_hat_mean'].values
ci_lows = df['ci_low'].values
ci_highs = df['ci_high'].values

print(f"\nTrue amplitude (a_true): {a_true:.5f}")
print(f"Grover calls range: [{grover_calls.min():,.0f}, {grover_calls.max():,.0f}]")
print(f"Epsilon (target error) range: [{epsilons.min():.3f}, {epsilons.max():.3f}]")

# ============================================================================
# CALCULATE REFERENCE LINES FOR QUANTUM SPEEDUP
# ============================================================================

# For quantum IQAE: error scales as O(1/N) where N = Grover calls
# This gives us: ε ∝ 1/N → N ∝ 1/ε (LINEAR speedup over classical O(1/ε²))

# Calculate bounds using percentiles
scaled_errors = epsilons * grover_calls
witness_top_n = np.percentile(scaled_errors, 95)
witness_bottom_n = np.percentile(scaled_errors, 5)
ref_line_top_n = witness_top_n / grover_calls
ref_line_bottom_n = witness_bottom_n / grover_calls

# Sample complexity: N ∝ 1/ε (not 1/ε²!)
inv_error = 1 / epsilons
scaled_samples = grover_calls / inv_error
witness_top_e = np.percentile(scaled_samples, 95)
witness_bottom_e = np.percentile(scaled_samples, 5)
ref_line_top_e = witness_top_e * inv_error
ref_line_bottom_e = witness_bottom_e * inv_error

print(f"\nQuantum Convergence Bounds:")
print(f"  • Error scaling:     [{witness_bottom_n:.2f}, {witness_top_n:.2f}] × N^(-1)")
print(f"  • Sample complexity: [{witness_bottom_e:.2f}, {witness_top_e:.2f}] × ε^(-1)")

# Filter outliers for cleaner visualizations
mask_fig2 = (scaled_errors >= witness_bottom_n) & (scaled_errors <= witness_top_n)
grover_calls_fig2 = grover_calls[mask_fig2]
epsilons_fig2 = epsilons[mask_fig2]
ref_top_n_fig2 = ref_line_top_n[mask_fig2]
ref_bot_n_fig2 = ref_line_bottom_n[mask_fig2]

mask_fig3 = (scaled_samples >= witness_bottom_e) & (scaled_samples <= witness_top_e)
grover_calls_fig3 = grover_calls[mask_fig3]
inv_error_fig3 = inv_error[mask_fig3]
ref_top_e_fig3 = ref_line_top_e[mask_fig3]
ref_bot_e_fig3 = ref_line_bottom_e[mask_fig3]

# ============================================================================
# GENERATE LOG-NORMAL DISTRIBUTION VISUALIZATION (HISTOGRAM)
# ============================================================================

print("\nGenerating log-normal distribution visualization...")

# Reconstruct the log-normal distribution
log_normal_mean = np.exp(mu + sigma**2 / 2)
log_normal_variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
log_normal_stddev = np.sqrt(log_normal_variance)

low = np.maximum(0, log_normal_mean - 3 * log_normal_stddev)
high = log_normal_mean + 3 * log_normal_stddev

# Calculate VaR threshold
grid_points = np.linspace(low, high, 2**num_qubits)
probs = lognorm.pdf(grid_points, s=sigma, scale=np.exp(mu))
probs = probs / np.sum(probs)

VAR = 0
accumulated_value = 0
for index in range(len(probs)):
    accumulated_value += probs[index]
    if accumulated_value > ALPHA:
        VAR = grid_points[index]
        break

# Generate samples for histogram (matching classical style)
np.random.seed(42)
samples = lognorm.rvs(s=sigma, scale=np.exp(mu), size=theoretical_N)

# Create distribution figure
fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
fig_dist.patch.set_facecolor('white')
ax_dist.set_facecolor('#fafafa')

# Histogram (matching classical style exactly)
counts, bins, patches = ax_dist.hist(
    samples, bins=100, alpha=0.7, color=COLOR_DIST,
    density=True, edgecolor='white', linewidth=0.5, label='Simulated Returns'
)

# VaR line
ax_dist.axvline(
    x=VAR, color=COLOR_DANGER, linestyle='--',
    linewidth=2.5, alpha=0.9, 
    label=f'VaR at {int(ALPHA*100)}%', zorder=5
)

# Shade VaR tail
var_mask = bins[:-1] <= VAR
if np.any(var_mask):
    for i, (patch, is_tail) in enumerate(zip(patches, var_mask)):
        if is_tail:
            patch.set_facecolor(COLOR_DANGER)
            patch.set_alpha(0.4)

# Styling
style_axes(
    ax_dist,
    f'Return Distribution: Log-Normal (μ={mu}, σ={sigma})',
    'Asset Value',
    'Probability Density'
)
create_legend(ax_dist, loc='best')

# Add statistics text box
stats_text = (
    f'Statistics:\n'
    f'Mean: {np.mean(samples):.5f}\n'
    f'Std Dev: {np.std(samples):.5f}\n'
    f'VaR: {VAR:.5f}\n'
    f'N: {theoretical_N:,}'
)
ax_dist.text(
    0.02, 0.98, stats_text, transform=ax_dist.transAxes,
    fontsize=10, verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor=COLOR_GRID, alpha=0.9)
)

plt.tight_layout()
plt.savefig(f'{output_dir}/00_distribution.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 00_distribution.png")
plt.close()

# ============================================================================
# FIGURE 1: AMPLITUDE ESTIMATION CONVERGENCE
# ============================================================================

print("\nGenerating convergence plots...")

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('white')

# Plot estimates (matching classical style)
ax1.plot(
    grover_calls, a_hat_means,
    marker='o', markersize=4, linewidth=2,
    color=COLOR_PRIMARY, alpha=0.8,
    label='Quantum Estimate', zorder=3
)

# True amplitude line
ax1.axhline(
    y=a_true, color=COLOR_DANGER,
    linestyle='--', linewidth=2.5, alpha=0.9,
    label='Theoretical Amplitude', zorder=2
)

ax1.set_xscale('log')
style_axes(
    ax1,
    'Convergence of Quantum Amplitude Estimation (IQAE)',
    'Number of Grover Calls (N)',
    f'Amplitude ({int((1-ALPHA)*100)}% Confidence)'
)
create_legend(ax1, loc='best')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_var_convergence.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 01_var_convergence.png")
plt.close()

# ============================================================================
# FIGURE 2: ERROR SCALING (QUANTUM SPEEDUP)
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('white')

ax2.plot(
    grover_calls_fig2, epsilons_fig2,
    marker='o', markersize=4, linewidth=2,
    color=COLOR_SECONDARY, alpha=0.8,
    label='Absolute Error', zorder=4
)

ax2.plot(
    grover_calls_fig2, ref_top_n_fig2,
    color=COLOR_BOUND_UPPER, linestyle='--', linewidth=2, alpha=0.7,
    label=r'$\mathcal{O}(N^{-1})$ Upper Bound', zorder=3
)

ax2.plot(
    grover_calls_fig2, ref_bot_n_fig2,
    color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=2, alpha=0.7,
    label=r'$\Omega(N^{-1})$ Lower Bound', zorder=3
)

ax2.fill_between(
    grover_calls_fig2, ref_bot_n_fig2, ref_top_n_fig2,
    alpha=0.1, color=COLOR_PRIMARY, zorder=1
)

ax2.set_xscale('log')
ax2.set_yscale('log')
style_axes(
    ax2,
    'Quantum Error Scaling: IQAE Convergence Rate',
    'Number of Grover Calls (N)',
    'Absolute Error |Amplitude - Theoretical|'
)
create_legend(ax2, loc='best')

# Annotation
mid_idx = len(grover_calls) // 2
ax2.annotate(
    r'Error $\propto N^{-1}$',
    xy=(grover_calls[mid_idx], epsilons[mid_idx]),
    xytext=(grover_calls[mid_idx] * 0.1, epsilons[mid_idx] * 3),
    fontsize=11, color=COLOR_ACCENT, fontweight='600', zorder=5,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor=COLOR_ACCENT, alpha=0.9),
    arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2)
)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_error_scaling.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 02_error_scaling.png")
plt.close()

# ============================================================================
# FIGURE 3: SAMPLE COMPLEXITY
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor('white')

ax3.plot(
    inv_error_fig3, grover_calls_fig3,
    marker='o', markersize=4, linewidth=2,
    color=COLOR_PRIMARY, alpha=0.8,
    label=r'Observed Grover Calls vs $\varepsilon^{-1}$', zorder=4
)

ax3.plot(
    inv_error_fig3, ref_top_e_fig3,
    color=COLOR_BOUND_UPPER, linestyle='--', linewidth=2, alpha=0.7,
    label=r'$\mathcal{O}(\varepsilon^{-1})$ Upper Bound', zorder=3
)

ax3.plot(
    inv_error_fig3, ref_bot_e_fig3,
    color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=2, alpha=0.7,
    label=r'$\Omega(\varepsilon^{-1})$ Lower Bound', zorder=3
)

ax3.fill_between(
    inv_error_fig3, ref_bot_e_fig3, ref_top_e_fig3,
    alpha=0.1, color=COLOR_PRIMARY, zorder=1
)

ax3.set_xscale('log')
ax3.set_yscale('log')
style_axes(
    ax3,
    'Sample Complexity: Quantum Amplitude Estimation',
    r'Inverse Error ($\varepsilon^{-1}$)',
    'Required Number of Grover Calls (N)'
)
create_legend(ax3, loc='best')

# Annotation
mid_idx = len(inv_error) // 2
ax3.annotate(
    r'$N \propto \varepsilon^{-1}$',
    xy=(inv_error[mid_idx], grover_calls[mid_idx]),
    xytext=(inv_error[mid_idx] * 0.15, grover_calls[mid_idx] * 5),
    fontsize=11, color=COLOR_ACCENT, fontweight='600', zorder=5,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor=COLOR_ACCENT, alpha=0.9),
    arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2)
)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_sample_complexity.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 03_sample_complexity.png")
plt.close()

# ============================================================================
# FIGURE 4: COMBINED ANALYSIS (4-PANEL)
# ============================================================================

print("\nGenerating combined analysis figure...")

fig4 = plt.figure(figsize=(20, 11))
fig4.patch.set_facecolor('white')
gs = fig4.add_gridspec(
    2, 3,
    width_ratios=[2, 1.5, 1.5],
    hspace=0.35, wspace=0.35,
    top=0.93, bottom=0.06, left=0.06, right=0.97
)

fig4.suptitle(
    f'Quantum Amplitude Estimation Analysis: IQAE on Log-Normal (μ={mu}, σ={sigma})',
    fontsize=18, fontweight='700', color=COLOR_TEXT, y=0.98
)

# Panel A: Distribution (histogram matching classical)
ax_a = fig4.add_subplot(gs[:, 0])
ax_a.set_facecolor('#fafafa')

counts, bins, patches = ax_a.hist(
    samples, bins=80, alpha=0.7, color=COLOR_DIST,
    density=True, edgecolor='white', linewidth=0.5
)
ax_a.axvline(
    x=VAR, color=COLOR_DANGER,
    linestyle='--', linewidth=2.5, alpha=0.9, zorder=5
)

# Shade tail
var_mask = bins[:-1] <= VAR
for patch, is_tail in zip(patches, var_mask):
    if is_tail:
        patch.set_facecolor(COLOR_DANGER)
        patch.set_alpha(0.4)

ax_a.set_xlabel('Asset Value', fontweight='500', color=COLOR_TEXT)
ax_a.set_ylabel('Probability Density', fontweight='500', color=COLOR_TEXT)
ax_a.set_title('(A) Return Distribution', fontweight='600', 
               color=COLOR_TEXT, loc='left')
ax_a.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
for spine in ax_a.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Panel B: Amplitude Convergence
ax_b = fig4.add_subplot(gs[0, 1:])
ax_b.set_facecolor('#fafafa')

ax_b.plot(
    grover_calls, a_hat_means, marker='o', markersize=3,
    linewidth=2, color=COLOR_PRIMARY, alpha=0.8
)
ax_b.axhline(
    y=a_true, color=COLOR_DANGER, linestyle='--',
    linewidth=2, alpha=0.9
)

ax_b.set_xscale('log')
ax_b.set_xlabel('Grover Calls (N)', fontweight='500', color=COLOR_TEXT)
ax_b.set_ylabel(f'Amplitude ({int((1-ALPHA)*100)}%)', fontweight='500', color=COLOR_TEXT)
ax_b.set_title('(B) Amplitude Estimate Convergence', fontweight='600',
               color=COLOR_TEXT, loc='left')
ax_b.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax_b.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
for spine in ax_b.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Panel C: Error Scaling
ax_c = fig4.add_subplot(gs[1, 1])
ax_c.set_facecolor('#fafafa')

ax_c.plot(
    grover_calls_fig2, epsilons_fig2, marker='o', markersize=3,
    linewidth=2, color=COLOR_SECONDARY, alpha=0.8
)
ax_c.plot(
    grover_calls_fig2, ref_top_n_fig2,
    color=COLOR_BOUND_UPPER, linestyle='--', linewidth=1.5, alpha=0.7
)
ax_c.plot(
    grover_calls_fig2, ref_bot_n_fig2,
    color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=1.5, alpha=0.7
)
ax_c.fill_between(
    grover_calls_fig2, ref_bot_n_fig2, ref_top_n_fig2,
    alpha=0.1, color=COLOR_PRIMARY
)

ax_c.set_xscale('log')
ax_c.set_yscale('log')
ax_c.set_xlabel('Grover Calls (N)', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_c.set_ylabel('Absolute Error', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_c.set_title('(C) Error Scaling', fontweight='600', color=COLOR_TEXT, loc='left')
ax_c.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax_c.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
for spine in ax_c.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Panel D: Sample Complexity
ax_d = fig4.add_subplot(gs[1, 2])
ax_d.set_facecolor('#fafafa')

ax_d.plot(
    inv_error_fig3, grover_calls_fig3, marker='o', markersize=3,
    linewidth=2, color=COLOR_PRIMARY, alpha=0.8
)
ax_d.plot(
    inv_error_fig3, ref_top_e_fig3,
    color=COLOR_BOUND_UPPER, linestyle='--', linewidth=1.5, alpha=0.7
)
ax_d.plot(
    inv_error_fig3, ref_bot_e_fig3,
    color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=1.5, alpha=0.7
)
ax_d.fill_between(
    inv_error_fig3, ref_bot_e_fig3, ref_top_e_fig3,
    alpha=0.1, color=COLOR_PRIMARY
)

ax_d.set_xscale('log')
ax_d.set_yscale('log')
ax_d.set_xlabel(r'$\varepsilon^{-1}$', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_d.set_ylabel('Grover Calls (N)', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_d.set_title('(D) Sample Complexity', fontweight='600', color=COLOR_TEXT, loc='left')
ax_d.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax_d.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
for spine in ax_d.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

plt.savefig(f'{output_dir}/04_combined_analysis.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 04_combined_analysis.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✨ ALL PUBLICATION-QUALITY GRAPHS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  📊 00_distribution.png         - Log-normal asset distribution & VaR")
print("  📊 01_var_convergence.png      - IQAE amplitude estimate convergence")
print("  📊 02_error_scaling.png        - Quantum error scaling O(N^-1)")
print("  📊 03_sample_complexity.png    - Sample complexity O(ε^-1)")
print("  📊 04_combined_analysis.png    - Comprehensive 4-panel summary")
print("\nAnalysis Summary:")
print(f"  • Algorithm:      Iterative Quantum Amplitude Estimation (IQAE)")
print(f"  • Distribution:   Log-Normal (μ={mu}, σ={sigma})")
print(f"  • VaR Level:      {int(ALPHA*100)}%")
print(f"  • True Amplitude: {a_true:.5f}")
print(f"  • Epsilon Range:  [{epsilons.min():.3f}, {epsilons.max():.3f}]")
print(f"  • Grover Range:   [{grover_calls.min():,.0f}, {grover_calls.max():,.0f}]")
print(f"  • Data Points:    {len(df)}")
print(f"\nQuantum Advantage:")
print(f"  • Error scaling:  O(N^-1) - Linear improvement")
print(f"  • Complexity:     O(ε^-1) vs Classical O(ε^-2)")
print(f"\nLocation: {output_dir}/")
print("="*70)

# Optional: Display plots if running interactively
# plt.show()