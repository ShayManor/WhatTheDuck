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

# Multi-day and distribution settings
T = 5                          # Number of days for multi-day VaR
dist = "gaussian"              # Distribution: "gaussian", "student-t", "skewnorm"
df = 3                         # Degrees of freedom for Student-t
skew_alpha = 7.0               # Skew parameter for skew-normal
rho = 0.0                      # AR(1) correlation coefficient


SAMPLE_COUNT = 5
SAMPLE_DAYS = 30
OUTPUT = "output/walk_days.png"
TRUE_MEAN_SAMPLES = 10**5

# ============================================================================
# SAMPLE POINT
# ============================================================================

def sample_return(
    mu: float,
    sigma: float,
    T: int = 1,
    dist: str = "gaussian",
    df: int = 5,
    skew_alpha: float = 0.0,
    rho: float = 0.0
) -> float:
    """
    Draw a single T-day return from the specified distribution and process.

    Returns:
        Single scalar total return
    """

    # Generate daily returns
    if dist == "gaussian":
        daily_returns = np.random.normal(mu, sigma, size=T)
    elif dist == "student-t":
        daily_returns = mu + sigma * np.random.standard_t(df, size=T)
    elif dist == "skewnorm":
        daily_returns = mu + sigma * skewnorm.rvs(skew_alpha, size=T)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    # Apply AR(1) correlation if requested
    if rho != 0.0 and T > 1:
        correlated = np.zeros_like(daily_returns)
        correlated[0] = daily_returns[0]
        for t in range(1, T):
            innovation = (daily_returns[t] - mu) * np.sqrt(1 - rho**2)
            correlated[t] = mu + rho * (correlated[t-1] - mu) + innovation
        daily_returns = correlated

    # Aggregate multi-day return
    return float(np.sum(daily_returns))


# ============================================================================
# SAMPLE ENGINE
# ============================================================================

sample_paths = []
for s in range(SAMPLE_COUNT):
    path = [0.0]
    for d in range(SAMPLE_DAYS):
        pt = sample_return(mu, sigma, T, dist, df, skew_alpha, rho)
        previous_pt = path[-1]
        pt += previous_pt
        path.append(pt)
    sample_paths.append(path)

days = [d for d in range(SAMPLE_DAYS + 1)]

raw_datas = [sample_return(mu, sigma, T, dist, df, skew_alpha, rho) for _ in range(TRUE_MEAN_SAMPLES)]
true_mean = sum(raw_datas) / len(raw_datas)
path_mean = [true_mean * d for d in range(SAMPLE_DAYS + 1)]

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


colors = plt.cm.viridis(np.linspace(0, 1, SAMPLE_COUNT))
for path, color in zip(sample_paths, colors):
    ax1.plot(
        days, path,
        marker='o', markersize=4, linewidth=2,
        color=color, alpha=0.6,
        zorder=3
    )
ax1.plot(
    days, path_mean,
    linestyle='--', linewidth=3,
    color=COLOR_DANGER, alpha=0.9,
    label='Theoretical Mean Path', zorder=4
)

style_axes(
    fig1,
    ax1,
    f'Example P&L Walks over Multiple Days',
    get_distribution_name(mu, sigma, dist, T, rho, df, skew_alpha),
    'Days',
    'Total Return'
)
create_legend(ax1, loc='best')

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {OUTPUT}")
plt.close()