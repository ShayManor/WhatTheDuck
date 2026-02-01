"""
MONTE CARLO VaR ANALYSIS - PUBLICATION-QUALITY VISUALIZATION
============================================================
Advanced Monte Carlo simulation supporting:
- Multi-day horizons
- Non-Gaussian distributions (Normal, Student-t, Skew-Normal)
- Temporal correlation (AR(1) process)
- Comprehensive convergence analysis

Author: Classical Monte Carlo Analysis
Date: 2026
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skewnorm
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

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
plt.rcParams['font.family'] = 'DejaVu Sans'

# Cyberpunk-inspired palette (dark + neon)
COLOR_BG = '#0b1020'
COLOR_PANEL = '#0f172a'
COLOR_GRID = '#1f2a44'
COLOR_TEXT = '#e6f1ff'
COLOR_MUTED = '#94a3b8'
COLOR_PRIMARY = '#38bdf8'      # Cyan
COLOR_SECONDARY = '#a78bfa'    # Violet
COLOR_ACCENT = '#22d3ee'       # Teal
COLOR_DANGER = '#fb7185'       # Neon pink
COLOR_BOUND_UPPER = '#60a5fa'  # Sky blue
COLOR_BOUND_LOWER = '#c084fc'  # Purple
COLOR_DIST = '#38bdf8'         # Distribution color

BRAND_TITLE = 'What The Duck'
BRAND_SUBTITLE = 'Hackathon Project'

BG_CMAP = LinearSegmentedColormap.from_list(
    'duck_bg', ['#0b1020', '#0f172a', '#1b1f3b']
)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Market parameters
mu = 0.15                      # Mean daily return (15%)
sigma = 0.20                   # Daily volatility (20%)
confidence_level = 0.95        # VaR confidence level

# Multi-day and distribution settings
T = 5                          # Number of days for multi-day VaR
dist = "skewnorm"              # Distribution: "gaussian", "student-t", "skewnorm"
df = 3                         # Degrees of freedom for Student-t
skew_alpha = 7.0               # Skew parameter for skew-normal
rho = 0.6                      # AR(1) correlation coefficient

# Where modeling error is intentionally held fixed
# You do not mix models during convergence plots.

# Simulation settings
num_samples_max = 10**7        # Maximum samples
num_samples_count = 250        # Number of sample sizes to test
theoretical_N = num_samples_max * 2 # Samples for theoretical VaR estimation
theoretical_estimations = 10        # Averaging runs for theoretical VaR
CPU_WORKERS = 10               # Parallel workers

# Generate logarithmically spaced sample sizes
num_samples_list = np.unique(
    np.logspace(1, np.log10(num_samples_max), num_samples_count, dtype=int)
).tolist()

# Output directory
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# MONTE CARLO ENGINE
# ============================================================================

@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results."""
    num_samples: int
    var_estimate: float
    error: float


def monte_carlo_var(
    N: int,
    mu: float,
    sigma: float,
    confidence_level: float,
    theoretical_var: float,
    T: int = 1,
    dist: str = "gaussian",
    df: int = 5,
    skew_alpha: float = 0.0,
    rho: float = 0.0
) -> MonteCarloResult:
    """
    Monte Carlo VaR estimation with advanced features.
    
    Parameters:
        N: Number of Monte Carlo scenarios
        mu: Mean daily return
        sigma: Daily volatility
        confidence_level: Confidence level for VaR
        theoretical_var: Theoretical VaR (for error calculation)
        T: Number of days in horizon
        dist: Distribution type ("gaussian", "student-t", "skewnorm")
        df: Degrees of freedom for Student-t
        skew_alpha: Skew parameter for skew-normal
        rho: AR(1) correlation coefficient
    
    Returns:
        MonteCarloResult with num_samples, var_estimate, and error
    """
    # Generate daily returns based on distribution
    if dist == "gaussian":
        daily_returns = np.random.normal(mu, sigma, size=(N, T))
    elif dist == "student-t":
        daily_returns = mu + sigma * np.random.standard_t(df, size=(N, T))
    elif dist == "skewnorm":
        daily_returns = mu + sigma * skewnorm.rvs(skew_alpha, size=(N, T))
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    
    # Apply AR(1) correlation if specified
    if rho != 0.0 and T > 1:
        correlated_returns = np.zeros_like(daily_returns)
        correlated_returns[:, 0] = daily_returns[:, 0]
        for t in range(1, T):
            innovation = (daily_returns[:, t] - mu) * np.sqrt(1 - rho**2)
            correlated_returns[:, t] = mu + rho * (correlated_returns[:, t-1] - mu) + innovation
        daily_returns = correlated_returns
    
    # Aggregate multi-day returns and compute VaR
    total_returns = daily_returns.sum(axis=1)
    # losses = -total_returns
    var_estimate = np.quantile(total_returns, 1 - confidence_level)
    error = abs(var_estimate - theoretical_var)
    
    return MonteCarloResult(N, var_estimate, error)


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def apply_background(ax):
    """Add a subtle tech-style gradient to the plotting area."""
    had_data = ax.has_data()
    xlim = ax.get_xlim() if had_data else None
    ylim = ax.get_ylim() if had_data else None
    ax.set_facecolor(COLOR_PANEL)
    ax.imshow(
        np.linspace(0, 1, 256).reshape(1, -1),
        extent=(0, 1, 0, 1),
        transform=ax.transAxes,
        cmap=BG_CMAP,
        aspect='auto',
        alpha=0.25,
        zorder=0
    )
    if had_data:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def add_branding(fig):
    """Add project branding to the figure."""
    fig.text(
        0.02, 0.98, BRAND_TITLE, ha='left', va='top',
        fontsize=14, color=COLOR_TEXT, fontweight='700'
    )
    fig.text(
        0.02, 0.945, BRAND_SUBTITLE, ha='left', va='top',
        fontsize=10, color=COLOR_MUTED
    )


def style_panel(ax):
    """Style ticks and spines for manual subplot layouts."""
    ax.tick_params(colors=COLOR_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
        spine.set_linewidth(1.2)


def glow_line(ax, x, y, color, lw=2.0, label=None, zorder=3, **kwargs):
    """Plot a line with a subtle neon glow."""
    line, = ax.plot(x, y, color=color, linewidth=lw, label=label, zorder=zorder, **kwargs)
    line.set_path_effects([
        pe.Stroke(linewidth=lw + 4, foreground=color, alpha=0.15),
        pe.Normal()
    ])
    return line


def style_axes(ax, title, xlabel, ylabel):
    """Apply consistent cyber styling to axes."""
    apply_background(ax)
    ax.set_title(title, fontweight='600', color=COLOR_TEXT, pad=18)
    ax.set_xlabel(xlabel, fontweight='500', color=COLOR_TEXT)
    ax.set_ylabel(ylabel, fontweight='500', color=COLOR_TEXT)
    ax.grid(True, which='major', linestyle='-', alpha=0.35, color=COLOR_GRID)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, color=COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
        spine.set_linewidth(1.2)


def create_legend(ax, **kwargs):
    """Create styled legend with consistent formatting."""
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      framealpha=0.95, edgecolor=COLOR_GRID, **kwargs)
    legend.get_frame().set_facecolor(COLOR_PANEL)
    return legend


def get_distribution_name(dist, df=None, skew_alpha=None):
    """Get formatted distribution name for labels."""
    if dist == "gaussian":
        return "Gaussian"
    elif dist == "student-t":
        return f"Student-t (df={df})"
    elif dist == "skewnorm":
        return f"Skew-Normal (α={skew_alpha})"
    return dist


# ============================================================================
# RUN MONTE CARLO SIMULATIONS
# ============================================================================

def main():
    global num_samples_list
    print("="*70)
    print("MONTE CARLO VaR ANALYSIS")
    print("="*70)
    print(f"\nParameters:")
    print(f"  • Mean return (μ):          {mu:.2%}")
    print(f"  • Volatility (σ):           {sigma:.2%}")
    print(f"  • Confidence level:         {confidence_level:.0%}")
    print(f"  • Time horizon:             {T} days")
    print(f"  • Distribution:             {get_distribution_name(dist, df, skew_alpha)}")
    print(f"  • Correlation (ρ):          {rho:.2f}")
    print(f"  • Sample range:             {num_samples_list[0]:,} to {num_samples_list[-1]:,}")
    print("\n" + "="*70)

    # Estimate theoretical VaR using large N and multiple runs
    print(f"\nEstimating theoretical VaR with N={theoretical_N:,} over {theoretical_estimations} runs...")
    theoretical_vars = []
    for i in range(theoretical_estimations):
        mc_run = monte_carlo_var(
            theoretical_N, mu, sigma, confidence_level, 0.0, T, dist, df, skew_alpha, rho
        )
        theoretical_vars.append(mc_run.var_estimate)
        print(f"  Run {i+1}/{theoretical_estimations}: VaR = {mc_run.var_estimate:.5f}")
    theoretical_var = float(np.mean(theoretical_vars))
    print(f"Theoretical VaR ({int(confidence_level*100)}%): {theoretical_var:.5f}")
    # This does two critical things:
    # It removes modeling error from the convergence study
    # It isolates probability estimation error only

    # Run parallel simulations
    print(f"\nRunning {len(num_samples_list)} simulations in parallel...")
    mc_results = []
    total_completed = 0

    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
        futures = [
            executor.submit(
                monte_carlo_var, N, mu, sigma, confidence_level, theoretical_var,
                T, dist, df, skew_alpha, rho
            )
            for N in num_samples_list
        ]
    
        for future in as_completed(futures):
            result = future.result()
            mc_results.append(result)
            total_completed += 1
            percent_complete = (total_completed / len(num_samples_list)) * 100
            print(f"  {percent_complete:5.1f}% | N={result.num_samples:>9,} | "
                  f"VaR={result.var_estimate:.5f} | Error={result.error:.3e}")

    # Sort results and extract data
    mc_results.sort(key=lambda x: x.num_samples)
    num_samples_list, var_results, errors = zip(*[
        (r.num_samples, r.var_estimate, r.error) for r in mc_results
    ])

    # ============================================================================
    # CALCULATE REFERENCE LINES AND FILTER OUTLIERS
    # ============================================================================

    # Calculate bounds using percentiles (more robust than min/max)
    scaled_errors = np.array(errors) * np.sqrt(np.array(num_samples_list))
    witness_top_n = np.percentile(scaled_errors, 95)
    witness_bottom_n = np.percentile(scaled_errors, 5)
    ref_line_top_n = witness_top_n / np.sqrt(num_samples_list)
    ref_line_bottom_n = witness_bottom_n / np.sqrt(num_samples_list)

    inv_error_sq = 1 / np.array(errors)**2
    num_samples_array = np.array(num_samples_list)
    scaled_samples = num_samples_array / inv_error_sq
    witness_top_e2 = np.percentile(scaled_samples, 95)
    witness_bottom_e2 = np.percentile(scaled_samples, 5)
    ref_line_top_e2 = witness_top_e2 * inv_error_sq
    ref_line_bottom_e2 = witness_bottom_e2 * inv_error_sq

    # Filter outliers for cleaner visualizations
    mask_fig2 = (scaled_errors >= witness_bottom_n) & (scaled_errors <= witness_top_n)
    num_samples_fig2 = num_samples_array[mask_fig2]
    errors_fig2 = np.array(errors)[mask_fig2]
    ref_top_n_fig2 = ref_line_top_n[mask_fig2]
    ref_bot_n_fig2 = ref_line_bottom_n[mask_fig2]

    mask_fig3 = (scaled_samples >= witness_bottom_e2) & (scaled_samples <= witness_top_e2)
    num_samples_fig3 = num_samples_array[mask_fig3]
    inv_error_sq_fig3 = inv_error_sq[mask_fig3]
    ref_top_e2_fig3 = ref_line_top_e2[mask_fig3]
    ref_bot_e2_fig3 = ref_line_bottom_e2[mask_fig3]

    print(f"\nConvergence Bounds:")
    print(f"  • Error scaling:     [{witness_bottom_n:.5f}, {witness_top_n:.5f}] × N^(-1/2)")
    print(f"  • Sample complexity: [{witness_bottom_e2:.5f}, {witness_top_e2:.5f}] × ε^(-2)")

    # Persist all data for later graphing
    csv_path = f"{output_dir}/mc_multi_day_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_samples",
                "var_estimate",
                "error",
                "scaled_error",
                "inv_error_sq",
                "theoretical_var",
                "confidence_level",
                "mu",
                "sigma",
                "T",
                "dist",
                "df",
                "skew_alpha",
                "rho",
            ],
        )
        writer.writeheader()
        for n, var_est, err in zip(num_samples_list, var_results, errors):
            inv_err_sq = np.inf if err == 0 else 1.0 / (err ** 2)
            writer.writerow(
                {
                    "num_samples": int(n),
                    "var_estimate": float(var_est),
                    "error": float(err),
                    "scaled_error": float(err * np.sqrt(n)),
                    "inv_error_sq": float(inv_err_sq),
                    "theoretical_var": float(theoretical_var),
                    "confidence_level": float(confidence_level),
                    "mu": float(mu),
                    "sigma": float(sigma),
                    "T": int(T),
                    "dist": dist,
                    "df": int(df),
                    "skew_alpha": float(skew_alpha),
                    "rho": float(rho),
                }
            )
    print(f"✓ Saved: {os.path.basename(csv_path)}")

    # ============================================================================
    # GENERATE DISTRIBUTION VISUALIZATION
    # ============================================================================

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
    fig_dist.patch.set_facecolor(COLOR_BG)
    add_branding(fig_dist)
    apply_background(ax_dist)

    # Histogram
    counts, bins, patches = ax_dist.hist(
        total_returns, bins=100, alpha=0.7, color=COLOR_DIST,
        density=True, edgecolor=COLOR_PANEL, linewidth=0.5, label='Simulated Returns'
    )

    # VaR line
    glow_line(
        ax_dist,
        np.array([var_for_plot, var_for_plot]),
        np.array([0, max(counts) * 1.05]),
        color=COLOR_DANGER,
        lw=2.5,
        label=f'VaR at {int(confidence_level*100)}%',
        zorder=5,
        linestyle='--',
        alpha=0.9
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
        ax_dist,
        f'Return Distribution: {get_distribution_name(dist, df, skew_alpha)} (T={T} days, ρ={rho})',
        f'{T}-Day Total Return',
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
        fontsize=10, verticalalignment='top', color=COLOR_TEXT,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PANEL,
                  edgecolor=COLOR_GRID, alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(f'{output_dir}/00_distribution.png',
                bbox_inches='tight', facecolor=COLOR_BG, edgecolor='none')
    print("✓ Saved: 00_distribution.png")
    plt.close()

    # ============================================================================
    # FIGURE 1: VaR CONVERGENCE
    # ============================================================================

    print("\nGenerating convergence plots...")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor(COLOR_BG)
    add_branding(fig1)

    glow_line(
        ax1,
        num_samples_list, var_results,
        color=COLOR_PRIMARY, lw=2.2,
        label='Monte Carlo Estimate', zorder=3,
        marker='o', markersize=4, alpha=0.9
    )
    glow_line(
        ax1,
        np.array([min(num_samples_list), max(num_samples_list)]),
        np.array([theoretical_var, theoretical_var]),
        color=COLOR_DANGER, lw=2.4,
        label='Theoretical VaR', zorder=2,
        linestyle='--', alpha=0.9
    )

    ax1.set_xscale('log')
    style_axes(
        ax1,
        'Convergence of Monte Carlo VaR Estimation',
        'Number of Samples (N)',
        f'Value-at-Risk ({int(confidence_level*100)}% Confidence)'
    )
    create_legend(ax1, loc='best')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_var_convergence.png',
                bbox_inches='tight', facecolor=COLOR_BG, edgecolor='none')
    print("✓ Saved: 01_var_convergence.png")
    plt.close()

    # ============================================================================
    # FIGURE 2: ERROR SCALING
    # ============================================================================

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor(COLOR_BG)
    add_branding(fig2)

    glow_line(
        ax2,
        num_samples_fig2, errors_fig2,
        color=COLOR_SECONDARY, lw=2.0,
        label='Absolute Error', zorder=4,
        marker='o', markersize=4, alpha=0.9
    )
    glow_line(
        ax2,
        num_samples_fig2, ref_top_n_fig2,
        color=COLOR_BOUND_UPPER, lw=2.0,
        label=r'$\mathcal{O}(N^{-1/2})$ Upper', zorder=3,
        linestyle='--', alpha=0.8
    )
    glow_line(
        ax2,
        num_samples_fig2, ref_bot_n_fig2,
        color=COLOR_BOUND_LOWER, lw=2.0,
        label=r'$\Omega(N^{-1/2})$ Lower', zorder=3,
        linestyle='-.', alpha=0.8
    )
    ax2.fill_between(
        num_samples_fig2, ref_bot_n_fig2, ref_top_n_fig2,
        alpha=0.12, color=COLOR_PRIMARY, zorder=1
    )

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    style_axes(
        ax2,
        'Monte Carlo Error Scaling: Classical Convergence Rate',
        'Number of Samples (N)',
        'Absolute Error |VaR - Theoretical|'
    )
    create_legend(ax2, loc='best')

    # Annotation
    mid_idx = len(num_samples_list) // 2
    ax2.annotate(
        r'Error $\propto N^{-1/2}$',
        xy=(num_samples_list[mid_idx], errors[mid_idx]),
        xytext=(num_samples_list[mid_idx] * 0.1, errors[mid_idx] * 3),
        fontsize=11, color=COLOR_ACCENT, fontweight='600', zorder=5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PANEL,
                  edgecolor=COLOR_ACCENT, alpha=0.9),
        arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2)
    )

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_error_scaling.png',
                bbox_inches='tight', facecolor=COLOR_BG, edgecolor='none')
    print("✓ Saved: 02_error_scaling.png")
    plt.close()

    # ============================================================================
    # FIGURE 3: SAMPLE COMPLEXITY
    # ============================================================================

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor(COLOR_BG)
    add_branding(fig3)

    glow_line(
        ax3,
        inv_error_sq_fig3, num_samples_fig3,
        color=COLOR_PRIMARY, lw=2.0,
        label=r'Observed Samples vs $\varepsilon^{-2}$', zorder=4,
        marker='o', markersize=4, alpha=0.9
    )
    glow_line(
        ax3,
        inv_error_sq_fig3, ref_top_e2_fig3,
        color=COLOR_BOUND_UPPER, lw=2.0,
        label=r'$\mathcal{O}(\varepsilon^{-2})$ Upper', zorder=3,
        linestyle='--', alpha=0.8
    )
    glow_line(
        ax3,
        inv_error_sq_fig3, ref_bot_e2_fig3,
        color=COLOR_BOUND_LOWER, lw=2.0,
        label=r'$\Omega(\varepsilon^{-2})$ Lower', zorder=3,
        linestyle='-.', alpha=0.8
    )
    ax3.fill_between(
        inv_error_sq_fig3, ref_bot_e2_fig3, ref_top_e2_fig3,
        alpha=0.12, color=COLOR_PRIMARY, zorder=1
    )

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    style_axes(
        ax3,
        'Sample Complexity: Classical Monte Carlo Scaling',
        r'Inverse Squared Error ($\varepsilon^{-2}$)',
        'Required Number of Samples (N)'
    )
    create_legend(ax3, loc='best')

    # Annotation
    mid_idx = len(inv_error_sq) // 2
    ax3.annotate(
        r'$N \propto \varepsilon^{-2}$',
        xy=(inv_error_sq[mid_idx], num_samples_list[mid_idx]),
        xytext=(inv_error_sq[mid_idx] * 0.15, num_samples_list[mid_idx] * 5),
        fontsize=11, color=COLOR_ACCENT, fontweight='600', zorder=5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PANEL,
                  edgecolor=COLOR_ACCENT, alpha=0.9),
        arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2)
    )

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_sample_complexity.png',
                bbox_inches='tight', facecolor=COLOR_BG, edgecolor='none')
    print("✓ Saved: 03_sample_complexity.png")
    plt.close()

    # ============================================================================
    # FIGURE 4: COMBINED ANALYSIS (4-PANEL)
    # ============================================================================

    print("\nGenerating combined analysis figure...")

    fig4 = plt.figure(figsize=(20, 11))
    fig4.patch.set_facecolor(COLOR_BG)
    add_branding(fig4)
    gs = fig4.add_gridspec(
        2,
        3,
        width_ratios=[2, 1.5, 1.5], # 40%, 30 + 30 = 60%
        hspace=0.35,
        wspace=0.35,
        top=0.93, bottom=0.06, left=0.06, right=0.97
    )

    fig4.suptitle(
        f'Classical Monte Carlo Analysis: {get_distribution_name(dist, df, skew_alpha)} '
        f'VaR (T={T}, ρ={rho})',
        fontsize=18, fontweight='700', color=COLOR_TEXT, y=0.98
    )

    # Panel A: Distribution (spans left column)
    ax_a = fig4.add_subplot(gs[:, 0])
    apply_background(ax_a)

    counts, bins, patches = ax_a.hist(
        total_returns, bins=80, alpha=0.7, color=COLOR_DIST,
        density=True, edgecolor=COLOR_PANEL, linewidth=0.5
    )
    glow_line(
        ax_a,
        np.array([var_for_plot, var_for_plot]),
        np.array([0, max(counts) * 1.05]),
        color=COLOR_DANGER, lw=2.5, zorder=5,
        linestyle='--', alpha=0.9
    )

    # Shade tail
    var_mask = bins[:-1] <= var_for_plot
    for patch, is_tail in zip(patches, var_mask):
        if is_tail:
            patch.set_facecolor(COLOR_DANGER)
            patch.set_alpha(0.4)

    ax_a.set_xlabel(f'{T}-Day Total Return', fontweight='500', color=COLOR_TEXT)
    ax_a.set_ylabel('Probability Density', fontweight='500', color=COLOR_TEXT)
    ax_a.set_title('(A) Return Distribution', fontweight='600', color=COLOR_TEXT, loc='left')
    ax_a.grid(True, which='major', linestyle='-', alpha=0.35, color=COLOR_GRID)
    style_panel(ax_a)

    # Panel B: VaR Convergence
    ax_b = fig4.add_subplot(gs[0, 1:])
    apply_background(ax_b)
    glow_line(
        ax_b,
        num_samples_list, var_results,
        color=COLOR_PRIMARY, lw=2.0, alpha=0.9,
        marker='o', markersize=3
    )
    glow_line(
        ax_b,
        np.array([min(num_samples_list), max(num_samples_list)]),
        np.array([theoretical_var, theoretical_var]),
        color=COLOR_DANGER, lw=2.0,
        linestyle='--', alpha=0.9
    )
    ax_b.set_xscale('log')
    ax_b.set_xlabel('Number of Samples (N)', fontweight='500', color=COLOR_TEXT)
    ax_b.set_ylabel(f'VaR ({int(confidence_level*100)}%)', fontweight='500', color=COLOR_TEXT)
    ax_b.set_title('(B) VaR Estimate Convergence', fontweight='600', color=COLOR_TEXT, loc='left')
    ax_b.grid(True, which='major', linestyle='-', alpha=0.35, color=COLOR_GRID)
    ax_b.grid(True, which='minor', linestyle=':', alpha=0.2, color=COLOR_GRID)
    style_panel(ax_b)

    # Panel C: Error Scaling
    ax_c = fig4.add_subplot(gs[1, 1])
    apply_background(ax_c)
    glow_line(
        ax_c,
        num_samples_fig2, errors_fig2,
        color=COLOR_SECONDARY, lw=2.0, alpha=0.9,
        marker='o', markersize=3
    )
    glow_line(
        ax_c,
        num_samples_fig2, ref_top_n_fig2,
        color=COLOR_BOUND_UPPER, lw=1.7, alpha=0.8,
        linestyle='--'
    )
    glow_line(
        ax_c,
        num_samples_fig2, ref_bot_n_fig2,
        color=COLOR_BOUND_LOWER, lw=1.7, alpha=0.8,
        linestyle='-.'
    )
    ax_c.fill_between(
        num_samples_fig2, ref_bot_n_fig2, ref_top_n_fig2,
        alpha=0.12, color=COLOR_PRIMARY
    )
    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.set_xlabel('Samples (N)', fontweight='500', color=COLOR_TEXT, fontsize=10)
    ax_c.set_ylabel('Absolute Error', fontweight='500', color=COLOR_TEXT, fontsize=10)
    ax_c.set_title('(C) Error Scaling', fontweight='600', color=COLOR_TEXT, loc='left')
    ax_c.grid(True, which='major', linestyle='-', alpha=0.35, color=COLOR_GRID)
    ax_c.grid(True, which='minor', linestyle=':', alpha=0.2, color=COLOR_GRID)
    style_panel(ax_c)

    # Panel D: Sample Complexity
    ax_d = fig4.add_subplot(gs[1, 2])
    apply_background(ax_d)
    glow_line(
        ax_d,
        inv_error_sq_fig3, num_samples_fig3,
        color=COLOR_PRIMARY, lw=2.0, alpha=0.9,
        marker='o', markersize=3
    )
    glow_line(
        ax_d,
        inv_error_sq_fig3, ref_top_e2_fig3,
        color=COLOR_BOUND_UPPER, lw=1.7, alpha=0.8,
        linestyle='--'
    )
    glow_line(
        ax_d,
        inv_error_sq_fig3, ref_bot_e2_fig3,
        color=COLOR_BOUND_LOWER, lw=1.7, alpha=0.8,
        linestyle='-.'
    )
    ax_d.fill_between(
        inv_error_sq_fig3, ref_bot_e2_fig3, ref_top_e2_fig3,
        alpha=0.12, color=COLOR_PRIMARY
    )
    ax_d.set_xscale('log')
    ax_d.set_yscale('log')
    ax_d.set_xlabel(r'$\varepsilon^{-2}$', fontweight='500', color=COLOR_TEXT, fontsize=10)
    ax_d.set_ylabel('Samples (N)', fontweight='500', color=COLOR_TEXT, fontsize=10)
    ax_d.set_title('(D) Sample Complexity', fontweight='600', color=COLOR_TEXT, loc='left')
    ax_d.grid(True, which='major', linestyle='-', alpha=0.35, color=COLOR_GRID)
    ax_d.grid(True, which='minor', linestyle=':', alpha=0.2, color=COLOR_GRID)
    style_panel(ax_d)

    plt.savefig(f'{output_dir}/04_combined_analysis.png',
                bbox_inches='tight', facecolor=COLOR_BG, edgecolor='none')
    print("✓ Saved: 04_combined_analysis.png")
    plt.close()

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n" + "="*70)
    print("✨ ALL PUBLICATION-QUALITY GRAPHS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Files:")
    print("  📊 00_distribution.png         - Return distribution & VaR visualization")
    print("  📊 01_var_convergence.png      - VaR estimate convergence")
    print("  📊 02_error_scaling.png        - Error scaling with O(N^-1/2) bounds")
    print("  📊 03_sample_complexity.png    - Sample complexity O(ε^-2) analysis")
    print("  📊 04_combined_analysis.png    - Comprehensive 4-panel summary")
    print("\nAnalysis Summary:")
    print(f"  • Distribution:   {get_distribution_name(dist, df, skew_alpha)}")
    print(f"  • Time Horizon:   {T} days")
    print(f"  • Correlation:    ρ = {rho}")
    print(f"  • Theoretical VaR: {theoretical_var:.5f}")
    print(f"  • Data Points:    {len(mc_results)}")
    print(f"\nLocation: {output_dir}/")
    print("="*70)

    # Optional: Display plots if running interactively
    plt.show()

if __name__ == "__main__":
    main()
