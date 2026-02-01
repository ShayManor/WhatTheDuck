"""
Heatmap Visualization for VaR IQAE Parameter Sweep

Creates publication-quality heatmaps showing resource requirements (shots and Grover calls)
as a function of epsilon (IQAE precision) and confidence level (VaR alpha).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import os


def load_sweep_data(csv_path: str) -> pd.DataFrame:
    """Load and validate sweep results."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points from {csv_path}")
    print(f"Epsilon range: {df['epsilon'].min():.4f} to {df['epsilon'].max():.4f}")
    print(f"Confidence levels: {sorted(df['confidence_level'].unique())}")
    return df


def create_heatmap_data(df: pd.DataFrame, metric: str) -> tuple:
    """
    Pivot data for heatmap visualization.
    
    Args:
        df: DataFrame with sweep results
        metric: Column name to visualize ('shots' or 'grover_calls')
    
    Returns:
        pivot_df: Pivoted data for heatmap
        epsilon_vals: Sorted epsilon values
        confidence_vals: Sorted confidence level values
    """
    # Get unique sorted values
    epsilon_vals = sorted(df['epsilon'].unique(), reverse=True)  # High to low for better viz
    confidence_vals = sorted(df['confidence_level'].unique())
    
    # Pivot table
    pivot_df = df.pivot_table(
        values=metric,
        index='epsilon',
        columns='confidence_level',
        aggfunc='mean'
    )
    
    # Reindex to ensure proper ordering
    pivot_df = pivot_df.reindex(index=epsilon_vals, columns=confidence_vals)
    
    return pivot_df, epsilon_vals, confidence_vals


def create_professional_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_path: str,
    title: str = None,
    cmap: str = 'YlOrRd',
    use_log_scale: bool = True,
    figsize: tuple = (12, 8),
    dpi: int = 300
):
    """
    Create a publication-quality heatmap.
    
    Args:
        df: DataFrame with sweep results
        metric: 'shots' or 'grover_calls'
        output_path: Path to save figure
        title: Custom title (auto-generated if None)
        cmap: Colormap name
        use_log_scale: Use logarithmic color scale
        figsize: Figure size in inches
        dpi: Resolution for saved figure
    """
    # Prepare data
    pivot_df, epsilon_vals, confidence_vals = create_heatmap_data(df, metric)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("paper", font_scale=1.3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=150)  # Lower DPI for display
    
    # Determine color normalization
    if use_log_scale and pivot_df.min().min() > 0:
        norm = LogNorm(vmin=pivot_df.min().min(), vmax=pivot_df.max().max())
        fmt = '.0f'
    else:
        norm = None
        fmt = '.0f'
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        norm=norm,
        cbar_kws={
            'label': f'{metric.replace("_", " ").title()}',
            'pad': 0.02
        },
        linewidths=0.5,
        linecolor='white',
        square=False,
        ax=ax,
        robust=True
    )
    
    # Customize labels
    ax.set_xlabel('Confidence Level (1 - α)', fontsize=14, fontweight='bold')
    ax.set_ylabel('IQAE Precision (ε)', fontsize=14, fontweight='bold')
    
    # Format tick labels
    ax.set_xticklabels([f'{x:.2f}' for x in confidence_vals], rotation=0)
    ax.set_yticklabels([f'{y:.3f}' for y in epsilon_vals], rotation=0)
    
    # Title
    if title is None:
        metric_name = "Measurement Shots" if metric == "shots" else "Grover Operator Calls"
        title = f'Resource Requirements: {metric_name}\nvs IQAE Precision and VaR Confidence Level'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Tight layout
    plt.tight_layout()
    
    # Save high-resolution version
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved heatmap to {output_path}")
    
    plt.close()


def create_comparison_heatmaps(
    df: pd.DataFrame,
    output_dir: str = "results",
    dpi: int = 300
):
    """
    Create side-by-side comparison of shots and Grover calls.
    """
    # Prepare data
    shots_pivot, epsilon_vals, confidence_vals = create_heatmap_data(df, 'shots')
    grover_pivot, _, _ = create_heatmap_data(df, 'grover_calls')
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    
    # Common parameters
    cmap = 'YlOrRd'
    linewidths = 0.5
    fmt = '.0f'
    
    # Left plot: Shots
    shots_norm = LogNorm(vmin=shots_pivot.min().min(), vmax=shots_pivot.max().max())
    sns.heatmap(
        shots_pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        norm=shots_norm,
        cbar_kws={'label': 'Total Shots', 'pad': 0.02},
        linewidths=linewidths,
        linecolor='white',
        ax=ax1,
        robust=True
    )
    ax1.set_xlabel('Confidence Level (1 - α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('IQAE Precision (ε)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Measurement Shots Required', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticklabels([f'{x:.2f}' for x in confidence_vals], rotation=0)
    ax1.set_yticklabels([f'{y:.3f}' for y in epsilon_vals], rotation=0)
    
    # Right plot: Grover calls
    grover_norm = LogNorm(vmin=grover_pivot.min().min(), vmax=grover_pivot.max().max())
    sns.heatmap(
        grover_pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        norm=grover_norm,
        cbar_kws={'label': 'Grover Calls', 'pad': 0.02},
        linewidths=linewidths,
        linecolor='white',
        ax=ax2,
        robust=True
    )
    ax2.set_xlabel('Confidence Level (1 - α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('IQAE Precision (ε)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Grover Operator Calls Required', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticklabels([f'{x:.2f}' for x in confidence_vals], rotation=0)
    ax2.set_yticklabels([f'{y:.3f}' for y in epsilon_vals], rotation=0)
    
    # Overall title
    fig.suptitle(
        'IQAE Resource Scaling for Value at Risk Estimation',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "var_comparison_heatmaps.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved comparison heatmaps to {output_path}")
    
    plt.close()


def create_accuracy_heatmap(
    df: pd.DataFrame,
    output_path: str,
    dpi: int = 300
):
    """
    Create heatmap showing VaR prediction accuracy (error).
    """
    # Calculate absolute percentage error
    df_clean = df.dropna(subset=['VaR_theoretical', 'VaR_predicted'])
    df_clean['abs_pct_error'] = np.abs(
        (df_clean['VaR_predicted'] - df_clean['VaR_theoretical']) / df_clean['VaR_theoretical'] * 100
    )
    
    # Prepare data
    pivot_df, epsilon_vals, confidence_vals = create_heatmap_data(df_clean, 'abs_pct_error')
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Create heatmap with reversed colormap (green=good, red=bad)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Absolute % Error', 'pad': 0.02},
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        robust=True,
        vmin=0,
        vmax=min(pivot_df.max().max(), 20)  # Cap at 20% for better contrast
    )
    
    # Customize
    ax.set_xlabel('Confidence Level (1 - α)', fontsize=14, fontweight='bold')
    ax.set_ylabel('IQAE Precision (ε)', fontsize=14, fontweight='bold')
    ax.set_title(
        'VaR Prediction Accuracy\nAbsolute Percentage Error (%)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xticklabels([f'{x:.2f}' for x in confidence_vals], rotation=0)
    ax.set_yticklabels([f'{y:.3f}' for y in epsilon_vals], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved accuracy heatmap to {output_path}")
    
    plt.close()


def create_all_visualizations(
    csv_path: str = "results/var_sweep.csv",
    output_dir: str = "results",
    dpi: int = 300
):
    """
    Generate all heatmap visualizations for the presentation.
    
    Args:
        csv_path: Path to CSV file with sweep results
        output_dir: Directory to save output figures
        dpi: Resolution for saved figures
    """
    # Load data
    df = load_sweep_data(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Publication-Quality Heatmaps")
    print("="*60 + "\n")
    
    # 1. Individual heatmap: Shots
    print("Creating shots heatmap...")
    create_professional_heatmap(
        df,
        metric='shots',
        output_path=os.path.join(output_dir, "heatmap_shots.png"),
        cmap='YlOrRd',
        use_log_scale=True,
        dpi=dpi
    )
    
    # 2. Individual heatmap: Grover calls
    print("Creating Grover calls heatmap...")
    create_professional_heatmap(
        df,
        metric='grover_calls',
        output_path=os.path.join(output_dir, "heatmap_grover_calls.png"),
        cmap='YlOrRd',
        use_log_scale=True,
        dpi=dpi
    )
    
    # 3. Side-by-side comparison
    print("Creating comparison heatmaps...")
    create_comparison_heatmaps(df, output_dir=output_dir, dpi=dpi)
    
    # 4. Accuracy heatmap
    print("Creating accuracy heatmap...")
    create_accuracy_heatmap(
        df,
        output_path=os.path.join(output_dir, "heatmap_accuracy.png"),
        dpi=dpi
    )
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    print(f"\nShots:")
    print(f"  Min: {df['shots'].min():.0f}")
    print(f"  Max: {df['shots'].max():.0f}")
    print(f"  Mean: {df['shots'].mean():.0f}")
    
    print(f"\nGrover Calls:")
    print(f"  Min: {df['grover_calls'].min():.0f}")
    print(f"  Max: {df['grover_calls'].max():.0f}")
    print(f"  Mean: {df['grover_calls'].mean():.0f}")
    
    # Accuracy stats
    df_clean = df.dropna(subset=['VaR_theoretical', 'VaR_predicted'])
    if len(df_clean) > 0:
        abs_pct_error = np.abs(
            (df_clean['VaR_predicted'] - df_clean['VaR_theoretical']) / df_clean['VaR_theoretical'] * 100
        )
        print(f"\nVaR Prediction Error (%):")
        print(f"  Min: {abs_pct_error.min():.2f}%")
        print(f"  Max: {abs_pct_error.max():.2f}%")
        print(f"  Mean: {abs_pct_error.mean():.2f}%")
        print(f"  Median: {abs_pct_error.median():.2f}%")
    
    print("\n" + "="*60)
    print("✓ All visualizations complete!")
    print("="*60)


if __name__ == "__main__":
    # Generate all visualizations
    create_all_visualizations(
        csv_path="results/var_sweep.csv",
        output_dir="results",
        dpi=300  # High resolution for presentations
    )