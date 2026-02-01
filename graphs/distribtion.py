import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


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
# SIMULATION PARAMETERS
# ============================================================================

# Market parameters
mu = 0.15                      # Mean daily return (15%)
sigma = 0.20                   # Daily volatility (20%)
confidence_level = 0.95        # VaR confidence level

# Multi-day and distribution settings
T = 4                          # Number of days for multi-day VaR
dist = "gaussian"              # Distribution: "gaussian", "student-t", "skewnorm"
df = 3                         # Degrees of freedom for Student-t
skew_alpha = 7.0               # Skew parameter for skew-normal
rho = 0.3                      # AR(1) correlation coefficient

# Simulation settings
theoretical_N = 10**7 * 2 # Samples for theoretical VaR estimation


OUTPUT = "output/distribution.png"


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

print("\nGenerating distribution visualization...")

# Simulate large sample for distribution
if dist == "gaussian":
    daily_returns = np.random.normal(mu, sigma, size=(theoretical_N, T))
elif dist == "student-t":
    daily_returns = mu + sigma * np.random.standard_t(df, size=(theoretical_N, T))
elif dist == "skewnorm":
    daily_returns = mu + sigma * skewnorm.rvs(skew_alpha, size=(theoretical_N, T))

# Apply correlation
if rho != 0.0 and T > 1:
    correlated_returns = np.zeros_like(daily_returns)
    correlated_returns[:, 0] = daily_returns[:, 0]
    for t in range(1, T):
        innovation = (daily_returns[:, t] - mu) * np.sqrt(1 - rho**2)
        correlated_returns[:, t] = mu + rho * (correlated_returns[:, t-1] - mu) + innovation
    daily_returns = correlated_returns

total_returns = daily_returns.sum(axis=1)
# losses = -total_returns
var_for_plot = np.quantile(total_returns, 1 - confidence_level)

# Create distribution figure
fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
fig_dist.patch.set_facecolor('white')
ax_dist.set_facecolor('#fafafa')

# Histogram
counts, bins, patches = ax_dist.hist(
    total_returns, bins=100, alpha=0.7, color=COLOR_DIST,
    density=True, edgecolor='white', linewidth=0.5, label='Simulated Returns'
)

# VaR line
ax_dist.axvline(
    x=var_for_plot, color=COLOR_DANGER, linestyle='--',
    linewidth=2.5, alpha=0.9, label=f'VaR at {int(confidence_level*100)}%', zorder=5
)

# Shade VaR tail
var_mask = bins[:-1] <= var_for_plot
if np.any(var_mask):
    for i, (patch, is_tail) in enumerate(zip(patches, var_mask)):
        if is_tail:
            patch.set_facecolor(COLOR_DANGER)
            patch.set_alpha(0.4)

# Styling
style_axes(
    fig_dist,
    ax_dist,
    'Return Distribution',
    get_distribution_name(mu, sigma, dist, T, rho, df, skew_alpha),
    f'P&L Return',
    'Probability Density'
)
create_legend(ax_dist, loc='best')

# Add statistics text box
stats_text = (
    f'Statistics:\n'
    f'Mean: {np.mean(total_returns):.5f}\n'
    f'Std Dev: {np.std(total_returns):.5f}\n'
    f'VaR: {var_for_plot:.5f}\n'
    f'N: {theoretical_N:,}'
)
ax_dist.text(
    0.02, 0.98, stats_text, transform=ax_dist.transAxes,
    fontsize=10, verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor=COLOR_GRID, alpha=0.9)
)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {OUTPUT}")
plt.close()