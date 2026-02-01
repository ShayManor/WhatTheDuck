"""
QUANTUM vs CLASSICAL VaR COMPARISON GRAPHS
=========================================
Generates four comparison plots using:
- IQAE sweep outputs from value_at_risk.py (results/iqae_epsilon_sweep.csv)
- Classical convergence outputs from value_at_risk.py (results/classical_var_convergence.csv)

Graphs:
1) Error vs query budget (log-log)
2) Budget to hit target accuracy
3) Robustness / sensitivity (confidence level)
4) MC convergence with variance bands
"""

import os
import csv
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================================
# STYLE (use matplotlib defaults - no theming)
# ============================================================================

# ============================================================================
# VALUE-AT-RISK SETTINGS (match value_at_risk.py)
# ============================================================================

CSV = "../data8.csv"
OUTPUT = "./e_vs_n.png"



def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)



def add_trend_line(ax, N, eps, color, label_prefix):
    """
    Fit and plot a power-law trend line: eps ~ C * N^{-alpha}
    """
    mask = (N > 0) & (eps > 0)
    N = N[mask]
    eps = eps[mask]

    logN = np.log10(N)
    logE = np.log10(eps)

    slope, intercept = np.polyfit(logN, logE, 1)
    alpha = -slope
    C = 10**intercept

    N_fit = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
    eps_fit = C * N_fit**(-alpha)

    ax.plot(
        N_fit, eps_fit,
        linestyle="--",
        linewidth=2.5,
        color=color,
        alpha=0.9,
        label=f"{label_prefix} fit: $N^{{-{alpha:.2f}}}$",
        zorder=2
    )


def error_vs_budget_plot(
    mc_n: np.ndarray,
    mc_err: np.ndarray,
    iqae_budget: np.ndarray,
    iqae_err: np.ndarray,
    output_path: str,
):
    eps_floor = 1e-6
    mc_err = np.maximum(mc_err, eps_floor)
    iqae_err = np.maximum(iqae_err, eps_floor)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        mc_n, mc_err,
        lw=2.0,
        label="Classical MC (VaR error)",
        marker="o", markersize=4, alpha=0.9
    )
    ax.plot(
        iqae_budget, iqae_err,
        lw=2.0,
        label="IQAE (probability error)",
        marker="o", markersize=4, alpha=0.9
    )
    # ax.fill_between(
    #     iqae_budget, iqae_err_low, iqae_err_high,
    #     alpha=0.12, zorder=2, label="IQAE ±1σ (probability)"
    # )

    # Reference slope lines
    # x_line = np.array([min(mc_n.min(), iqae_budget.min()), max(mc_n.max(), iqae_budget.max())])
    # c_half = mc_err[0] * math.sqrt(mc_n[0])
    # c_one = iqae_err[0] * iqae_budget[0]
    # ax.plot(x_line, c_half / np.sqrt(x_line), "--", alpha=0.8, label="slope -1/2")
    # ax.plot(x_line, c_one / x_line, "-.", alpha=0.8, label="slope -1")

    add_trend_line(
        ax,
        mc_n,
        mc_err,
        color=ax.lines[0].get_color(),
        label_prefix="MC"
    )
    add_trend_line(
        ax,
        iqae_budget,
        iqae_err,
        color=ax.lines[1].get_color(),
        label_prefix="IQAE"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(
        ax,
        "Error Scaling: Classical vs Quantum",
        "Oracle Calls / Samples [log]",
        "Estimation Error [log]",
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    results = pd.read_csv(CSV)

    # Keep only relevant rows (dist = normal, alpha=0.1)
    dist = "normal"
    results = results[results['dist'] == dist]
    results = results[results['var_alpha'] == 0.05]
    
    # Split into MC and QC
    results_mc = results[results['method'] == 'classical'].copy()
    results_qc = results[results['method'] == 'quantum'].copy()

    # sort by queries
    results_mc = results_mc.sort_values(by='queries')
    results_qc = results_qc.sort_values(by='queries')


    error_vs_budget_plot(
        mc_n=results_mc['queries'].to_numpy(),
        mc_err=results_mc['error'].to_numpy(),
        iqae_budget=results_qc['queries'].to_numpy(),
        iqae_err=results_qc['epsilon'].to_numpy(),
        output_path=OUTPUT,
    )


if __name__ == "__main__":
    main()