import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


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

# Professional color palette
COLOR_TEXT = '#1f2937'         # Dark gray text
COLOR_GRID = '#e5e7eb'         # Light gray grid
COLOR_THEORETICAL = '#bf616a'
# COLOR_MC = '#5e81ac'
# COLOR_QC = '#bf616a'

# ============================================================================
# READ CSV RESULTS
# ============================================================================

# method,dist,queries,epsilon,p_hat,true_p,error
CSV = "../data2.csv"
OUTPUT = "./var_convergence.png"

results = pd.read_csv(CSV)

# Keep only relevant rows (dist = normal)
dist = "normal"
results = results[results['dist'] == dist]

# Split into MC and QC
results_mc = results[results['method'] == 'classical'].copy()
results_qc = results[results['method'] == 'quantum'].copy()

# sort by queries
results_mc = results_mc.sort_values(by='queries')
results_qc = results_qc.sort_values(by='queries')

# Extract parameters from the first row
theoretical_var = results_mc['true_p'].values[0]


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def style_axes(ax, title, xlabel, ylabel):
    """Apply consistent professional styling to axes."""
    ax.set_facecolor('#fafafa')
    ax.set_title(title, fontweight='600', color=COLOR_TEXT, pad=10)
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


# def get_distribution_name(mu, sigma, dist, T, rho, df, skew_alpha):
#     """Get formatted distribution name for labels."""
#     if dist == "gaussian":
#         return f"Gaussian - T={T}, ρ={rho}, μ={mu}, σ={sigma}"
#     elif dist == "student-t":
#         return f"Student-t - T={T}, ρ={rho}, μ={mu}, σ={sigma}, df={df}"
#     elif dist == "skewnorm":
#         return f"Skew-Normal - T={T}, ρ={rho}, μ={mu}, σ={sigma}, α={skew_alpha}"
#     return dist

# ============================================================================
# PLOT
# ============================================================================

# Make output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("\nGenerating plot...")

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('white')


ax1.plot(
    results_mc["queries"], results_mc["p_hat"],
    marker='o', markersize=4, linewidth=2,
    # color=COLOR_MC, alpha=0.8,
    label='Monte Carlo Estimate', zorder=3
)
ax1.plot(
	results_qc["queries"], results_qc["p_hat"],
	marker='o', markersize=4, linewidth=2,
	# color=COLOR_QC, alpha=0.8,
	label='Quantum Estimate', zorder=3
)
ax1.axhline(
    y=theoretical_var, color=COLOR_THEORETICAL,
    linestyle='--', linewidth=2.5, alpha=0.9,
    label='Theoretical VaR', zorder=2
)
ax1.set_xscale('log')

style_axes(
    ax1,
    'Convergence of Monte Carlo VaR Estimation',
    'Number of Samples (N) [log scale]',
    f'Value-at-Risk ({int(.95*100)}% Confidence)'
)
create_legend(ax1, loc='best')

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {OUTPUT}")
plt.close()