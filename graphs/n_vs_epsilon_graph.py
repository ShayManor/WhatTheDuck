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
COLOR_THEORETICAL = '#2e3440'
COLOR_MC = '#5e81ac'
COLOR_QC = '#bf616a'

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# # Market parameters
# mu = 0.15                      # Mean daily return (15%)
# sigma = 0.20                   # Daily volatility (20%)

# # Multi-day and distribution settings
# T = 5                          # Number of days for multi-day VaR
# dist = "gaussian"              # Distribution: "gaussian", "student-t", "skewnorm"
# df = 3                         # Degrees of freedom for Student-t
# skew_alpha = 7.0               # Skew parameter for skew-normal
# rho = 0.0                      # AR(1) correlation coefficient


# ============================================================================
# READ CSV RESULTS
# ============================================================================

# CSV_HEADERS = [
#     "Epsilon",
#     "N",
#     "VaR_prediction",
#     "VaR_theoretical",
#     "mu", "sigma", "confidence_level", "T", "dist", "df", "skew_alpha", "rho"
# ]

CSV_INPUT1 = "data/monte_carlo_naive.csv"
CSV_INPUT2 = "data/monte_carlo_naive2.csv"
OUTPUT = "output/n_vs_epsilon.png"

results_mc = pd.read_csv(CSV_INPUT1)
results_qc = pd.read_csv(CSV_INPUT2)

# Ensure all data uses the same distribution parameters
params = results_mc.iloc[0][["mu", "sigma", "confidence_level", "T", "dist", "df", "skew_alpha", "rho"]]
for col in params.index:
	assert all(results_mc[col] == params[col]), f"Mismatch in MC data for {col}"
	assert all(results_qc[col] == params[col]), f"Mismatch in QC data for {col}"
    
# Ensure theoretical VaR is consistent
# To be consistent it must be exactly the same within one file, but may differ by 2% between files
theoretical_var_mc = results_mc.iloc[0]["VaR_theoretical"]
theoretical_var_qc = results_qc.iloc[0]["VaR_theoretical"]
assert all(results_mc["VaR_theoretical"] == theoretical_var_mc), "Mismatch in MC theoretical VaR"
assert all(results_qc["VaR_theoretical"] == theoretical_var_qc), "Mismatch in QC theoretical VaR"
theoretical_var = (theoretical_var_mc + theoretical_var_qc) / 2
assert abs(theoretical_var_mc - theoretical_var_qc) / theoretical_var < 0.02, "Theoretical VaR differs by more than 2% between MC and QC"

mu = params["mu"]
sigma = params["sigma"]
confidence_level = params["confidence_level"]
T = params["T"]
dist = params["dist"]
df = params["df"]
skew_alpha = params["skew_alpha"]
rho = params["rho"]

# Sort by N
results_mc = results_mc.sort_values(by="N")
results_qc = results_qc.sort_values(by="N")

# ============================================================================
# TREND LINE
# ============================================================================

def add_trend_line(ax, N, eps, color, label_prefix):
    """
    Fit and plot a power-law trend line: eps ~ C * N^{-alpha}
    """
    # Remove non-positive values just in case
    mask = (N > 0) & (eps > 0)
    N = N[mask]
    eps = eps[mask]

    logN = np.log10(N)
    logE = np.log10(eps)

    # Linear fit in log-log space
    slope, intercept = np.polyfit(logN, logE, 1)
    alpha = -slope
    C = 10**intercept

    # Smooth line for plotting
    N_fit = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
    eps_fit = C * N_fit**(-alpha)

    ax.plot(
        N_fit, eps_fit,
        linestyle='-',
        linewidth=2.5,
        color=color,
        alpha=0.9,
        label=f"{label_prefix} trend: $N^{{-{alpha:.2f}}}$",
        zorder=2
    )


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def style_axes(fig, ax, title, subtitle, xlabel, ylabel):
    """Apply consistent professional styling to axes."""
    fig.suptitle(title, fontweight='700', color=COLOR_TEXT, y=.96)
    ax.set_facecolor('#fafafa')
    ax.set_title(subtitle, fontweight='300', color=COLOR_TEXT, pad=10)
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


def get_distribution_name(mu, sigma, dist, T, rho, df, skew_alpha):
    """Get formatted distribution name for labels."""
    if dist == "gaussian":
        return f"Gaussian - T={T}, ρ={rho}, μ={mu}, σ={sigma}"
    elif dist == "student-t":
        return f"Student-t - T={T}, ρ={rho}, μ={mu}, σ={sigma}, df={df}"
    elif dist == "skewnorm":
        return f"Skew-Normal - T={T}, ρ={rho}, μ={mu}, σ={sigma}, α={skew_alpha}"
    return dist

# ============================================================================
# PLOT
# ============================================================================

# Make output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("\nGenerating plot...")

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('white')


ax1.plot(
    results_mc["N"], results_mc["Epsilon"],
    marker='o', markersize=4, linewidth=2,
    color=COLOR_MC, alpha=0.8,
    label='Monte Carlo Estimate', zorder=3
)
ax1.plot(
	results_qc["N"], results_qc["Epsilon"],
	marker='o', markersize=4, linewidth=2,
	color=COLOR_QC, alpha=0.8,
	label='Quantum Estimate', zorder=3
)

add_trend_line(ax1, results_mc["N"], results_mc["Epsilon"], COLOR_MC, "Monte Carlo")
add_trend_line(ax1, results_qc["N"], results_qc["Epsilon"], COLOR_QC, "Quantum")

ax1.set_xscale('log')
ax1.set_yscale('log')

style_axes(
    fig1,
    ax1,
    'Error Scaling',
    get_distribution_name(mu, sigma, dist, T, rho, df, skew_alpha),
    'Number of Samples (N) [log scale]',
    f'Absolute Error in VaR Estimate [log scale]'
)
create_legend(ax1, loc='best')

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {OUTPUT}")
plt.close()