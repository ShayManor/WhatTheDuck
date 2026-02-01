"""
Heatmap Visualization for VaR IQAE Parameter Sweep

Creates publication-quality heatmap showing resource requirements (shots)
as a function of epsilon (IQAE precision) and alpha_iqae (stopping value tolerance).
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
    print(f"Alpha_iqae values: {sorted(df['alpha_iqae'].unique())}")
    return df


def create_heatmap_data(df: pd.DataFrame, metric: str) -> tuple:
    """
    Pivot data for heatmap visualization.
    
    Args:
        df: DataFrame with sweep results
        metric: Column name to visualize ('shots')
    
    Returns:
        pivot_df: Pivoted data for heatmap
        epsilon_vals: Sorted epsilon values
        alpha_iqae_vals: Sorted alpha_iqae values
    """
    # Get unique sorted values
    epsilon_vals = sorted(df['epsilon'].unique(), reverse=True)  # High to low for better viz
    alpha_iqae_vals = sorted(df['alpha_iqae'].unique())
    
    # Pivot table
    pivot_df = df.pivot_table(
        values=metric,
        index='epsilon',
        columns='alpha_iqae',
        aggfunc='mean'
    )
    
    # Reindex to ensure proper ordering
    pivot_df = pivot_df.reindex(index=epsilon_vals, columns=alpha_iqae_vals)
    
    return pivot_df, epsilon_vals, alpha_iqae_vals


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
        metric: 'shots'
        output_path: Path to save figure
        title: Custom title (auto-generated if None)
        cmap: Colormap name
        use_log_scale: Use logarithmic color scale
        figsize: Figure size in inches
        dpi: Resolution for saved figure
    """
    # Prepare data
    pivot_df, epsilon_vals, alpha_iqae_vals = create_heatmap_data(df, metric)
    
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
    ax.set_xlabel('IQAE Stopping Tolerance (α_IQAE)', fontsize=14, fontweight='bold')
    ax.set_ylabel('IQAE Precision (ε)', fontsize=14, fontweight='bold')
    
    # Format tick labels
    ax.set_xticklabels([f'{x:.3f}' for x in alpha_iqae_vals], rotation=0)
    ax.set_yticklabels([f'{y:.3f}' for y in epsilon_vals], rotation=0)
    
    # Title
    if title is None:
        metric_name = "Measurements"
        title = f'Resource Requirements: {metric_name}\nvs IQAE Precision and Stopping Tolerance'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Tight layout
    plt.tight_layout()
    
    # Save high-resolution version
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved heatmap to {output_path}")
    
    plt.close()


def create_all_visualizations(
    csv_path: str = "results/var_sweep_01.csv",
    output_dir: str = "results",
    dpi: int = 300
):
    """
    Generate shots heatmap visualization.
    
    Args:
        csv_path: Path to CSV file with sweep results
        output_dir: Directory to save output figures
        dpi: Resolution for saved figures
    """
    # Load data
    df = load_sweep_data(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Publication-Quality Heatmap")
    print("="*60 + "\n")
    
    # Generate shots heatmap
    print("Creating shots heatmap...")
    create_professional_heatmap(
        df,
        metric='shots',
        output_path=os.path.join(output_dir, "heatmap_shots.png"),
        cmap='YlOrRd',
        use_log_scale=True,
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
    print(f"  Median: {df['shots'].median():.0f}")
    
    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    # Generate visualization
    create_all_visualizations(
        csv_path="results/mc_sweep.csv",
        output_dir="results",
        dpi=300  # High resolution for presentations
    )