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
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['mathtext.fontset'] = 'stix'

# Professional color palette - different colors for each T
COLORS = {
    1: '#1e3a8a',  # Deep blue
    2: '#3b82f6',  # Bright blue
    3: '#6366f1',  # Indigo
    4: '#8b5cf6',  # Purple
    5: '#a855f7',  # Lighter purple
}
COLOR_GRID = '#e5e7eb'         # Light gray grid
COLOR_TEXT = '#1f2937'         # Dark gray text

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Market parameters
mu = 0.15                         # Mean daily return
sigma = 0.2                      # Daily volatility
confidence_level = 0.95        # VaR confidence level

# Distribution settings
T_values = [1, 2, 3, 4, 5]     # Time horizons to plot
dist = "gaussian"              # Distribution: "gaussian", "student-t", "skewnorm"
df = 3                         # Degrees of freedom for Student-t
skew_alpha = 7.0               # Skew parameter for skew-normal
rho = 0.5                      # AR(1) correlation coefficient (set to non-zero to see effect)

# Simulation settings
theoretical_N = 10**6          # Samples for theoretical VaR estimation

OUTPUT = "output/distribution_overlay.png"


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


def get_distribution_name(mu, sigma, dist, rho, df, skew_alpha):
    """Get formatted distribution name for labels."""
    rho_str = f", ρ={rho}" if rho != 0 else ""
    if dist == "gaussian":
        return f"Gaussian - μ={mu}, σ={sigma}{rho_str}"
    elif dist == "student-t":
        return f"Student-t - μ={mu}, σ={sigma}, df={df}{rho_str}"
    elif dist == "skewnorm":
        return f"Skew-Normal - μ={mu}, σ={sigma}, α={skew_alpha}{rho_str}"
    return dist


def simulate_returns(mu, sigma, dist, T, rho, theoretical_N, df=3, skew_alpha=7.0):
    """Simulate returns for a given time horizon with AR correlation."""
    # Generate base returns
    if dist == "gaussian":
        daily_returns = np.random.normal(mu, sigma, size=(theoretical_N, T))
    elif dist == "student-t":
        daily_returns = mu + sigma * np.random.standard_t(df, size=(theoretical_N, T))
    elif dist == "skewnorm":
        daily_returns = mu + sigma * skewnorm.rvs(skew_alpha, size=(theoretical_N, T))
    
    # Apply AR(1) correlation if rho != 0 and T > 1
    if rho != 0.0 and T > 1:
        correlated_returns = np.zeros_like(daily_returns)
        correlated_returns[:, 0] = daily_returns[:, 0]
        for t in range(1, T):
            innovation = (daily_returns[:, t] - mu) * np.sqrt(1 - rho**2)
            correlated_returns[:, t] = mu + rho * (correlated_returns[:, t-1] - mu) + innovation
        daily_returns = correlated_returns
    
    # Sum returns across time
    total_returns = daily_returns.sum(axis=1)
    return total_returns


# ============================================================================
# PLOT
# ============================================================================

# Make output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("\nGenerating overlaid distribution visualization...")

# Create distribution figure
fig_dist, ax_dist = plt.subplots(figsize=(12, 7))
fig_dist.patch.set_facecolor('white')
ax_dist.set_facecolor('#fafafa')

# Store all returns for determining plot limits
all_returns = []
var_lines = []

# Simulate and plot for each T value
for T in T_values:
    print(f"Simulating T={T}...")
    
    # Simulate returns
    total_returns = simulate_returns(
        mu, sigma, dist, T, rho, theoretical_N, df, skew_alpha
    )
    all_returns.append(total_returns)
    
    # Calculate VaR
    var_value = np.quantile(total_returns, 1 - confidence_level)
    var_lines.append((T, var_value))
    
    # Calculate opacity: increase opacity with T
    # T=1 -> alpha=0.3, T=5 -> alpha=0.7
    alpha_hist = 0.3 + (T - 1) * 0.1
    alpha_line = 0.5 + (T - 1) * 0.1
    
    # Plot histogram
    ax_dist.hist(
        total_returns, bins=200, alpha=alpha_hist, color=COLORS[T],
        density=True, edgecolor='none', 
        label=f'T={T} (VaR: {var_value:.3f})'
    )
    
    # Plot VaR line
    ax_dist.axvline(
        x=var_value, color=COLORS[T], linestyle='--',
        linewidth=2, alpha=alpha_line, zorder=5
    )

# Styling
subtitle = get_distribution_name(mu, sigma, dist, rho, df, skew_alpha)
style_axes(
    fig_dist,
    ax_dist,
    'Return Distribution Across Time Horizons',
    subtitle,
    'Cumulative P&L Return',
    'Probability Density'
)

# Create legend with VaR values
create_legend(ax_dist, loc='upper left', ncol=1)

# Add info text box
info_text = (
    f'95% VaR by Horizon:\n' +
    '\n'.join([f'T={T}: {var:.3f}' for T, var in var_lines])
)
ax_dist.text(
    0.98, 0.98, info_text, transform=ax_dist.transAxes,
    fontsize=9, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor=COLOR_GRID, alpha=0.95)
)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {OUTPUT}")
plt.close()

print("\nDone!")