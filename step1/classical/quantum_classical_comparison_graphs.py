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
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================================
# STYLE (use matplotlib defaults - no theming)
# ============================================================================

# ============================================================================
# VALUE-AT-RISK SETTINGS (match value_at_risk.py)
# ============================================================================

MU = 0.7
SIGMA = 0.13

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)


def load_iqae_results(path: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    budgets = []
    a_hat = []
    a_hat_std = []
    prob_err = []
    ci_low = []
    ci_high = []
    a_true = None
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if a_true is None:
                try:
                    a_true = float(row.get("a_true", "nan"))
                except ValueError:
                    a_true = None
            a_hat_mean = float(row["a_hat_mean"])
            a_hat_sigma = float(row.get("a_hat_std", 0.0) or 0.0)
            prob_err_mean = float(row.get("abs_error_mean", 0.0) or 0.0)
            ci_l = float(row.get("ci_low", 0.0) or 0.0)
            ci_h = float(row.get("ci_high", 0.0) or 0.0)
            grover_calls = float(row.get("grover_calls_mean", 0) or 0)
            shots = float(row.get("shots_total_mean", 0) or 0)
            budget = grover_calls if grover_calls > 0 else shots
            if budget <= 0:
                continue
            budgets.append(budget)
            a_hat.append(a_hat_mean)
            a_hat_std.append(a_hat_sigma)
            prob_err.append(prob_err_mean)
            ci_low.append(ci_l)
            ci_high.append(ci_h)
    if not budgets:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            float("nan"),
        )
    order = np.argsort(budgets)
    return (
        np.array(budgets)[order],
        np.array(a_hat)[order],
        np.array(a_hat_std)[order],
        np.array(prob_err)[order],
        np.array(ci_low)[order],
        np.array(ci_high)[order],
        a_true,
    )


def load_classical_convergence(path: str) -> Dict[int, Dict[str, float]]:
    by_n_err: Dict[int, List[float]] = {}
    by_n_var: Dict[int, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["sample_size"])
            err = float(row["abs_error"])
            var_est = float(row.get("var_estimate", 0.0))
            by_n_err.setdefault(n, []).append(err)
            by_n_var.setdefault(n, []).append(var_est)

    stats: Dict[int, Dict[str, float]] = {}
    for n, errs in by_n_err.items():
        arr = np.array(errs, dtype=float)
        var_arr = np.array(by_n_var.get(n, []), dtype=float)
        stats[n] = {
            "mean": float(np.mean(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "var_mean": float(np.mean(var_arr)) if var_arr.size else 0.0,
            "var_p10": float(np.percentile(var_arr, 10)) if var_arr.size else 0.0,
            "var_p90": float(np.percentile(var_arr, 90)) if var_arr.size else 0.0,
        }
    return stats



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
    iqae_err_low: np.ndarray,
    iqae_err_high: np.ndarray,
    output_path: str,
):
    eps_floor = 1e-6
    mc_err = np.maximum(mc_err, eps_floor)
    iqae_err = np.maximum(iqae_err, eps_floor)
    iqae_err_low = np.maximum(iqae_err_low, eps_floor)
    iqae_err_high = np.maximum(iqae_err_high, eps_floor)

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


def budget_to_target_plot(
    iqae_budget: np.ndarray,
    iqae_ci_low: np.ndarray,
    iqae_ci_high: np.ndarray,
    p_true: float,
    output_path: str,
):
    targets = np.array(
        [1e-3, 1.3e-3, 1.7e-3, 2.2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2],
        dtype=float,
    )

    def min_budget_for_target(budgets, errors, target):
        order = np.argsort(errors)
        errors_sorted = errors[order]
        budgets_sorted = budgets[order]
        idx = np.where(errors_sorted <= target)[0]
        if len(idx) == 0:
            return None
        return float(np.min(budgets_sorted[idx]))

    # MC CI half-width: z * sqrt(p(1-p)/N) <= eps => N >= z^2 * p(1-p) / eps^2
    z = 1.96
    mc_required = np.array(
        [math.ceil((z * z) * p_true * (1 - p_true) / (t ** 2)) for t in targets],
        dtype=float,
    )
    iqae_half_width = 0.5 * (iqae_ci_high - iqae_ci_low)
    iqae_required = [min_budget_for_target(iqae_budget, iqae_half_width, t) for t in targets]
    iqae_required = np.array([v if v is not None else np.nan for v in iqae_required], dtype=float)

    # Enforce monotonic non-increasing budget as epsilon increases.
    def monotone_envelope(values: np.ndarray) -> np.ndarray:
        out = values.copy()
        valid = np.isfinite(out)
        if not np.any(valid):
            return out
        last = np.inf
        for i in range(len(out)):
            if not np.isfinite(out[i]):
                continue
            last = min(last, out[i])
            out[i] = last
        return out

    mc_required = monotone_envelope(mc_required)
    iqae_required = monotone_envelope(iqae_required)
    mask = np.isfinite(iqae_required)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(
        targets, mc_required,
        where="post", linewidth=2.0, label="Classical MC (CI half-width)"
    )
    ax.scatter(targets, mc_required, s=24)
    ax.step(
        targets[mask], iqae_required[mask],
        where="post", linewidth=2.0, label="IQAE (CI half-width)"
    )
    ax.scatter(targets[mask], iqae_required[mask], s=24)

    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(
        ax,
        "Budget to Hit Target (CI Half-Width)",
        "Target εp (log)",
        "Required Queries (log)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def ci_halfwidth_vs_budget_plot(
    iqae_budget: np.ndarray,
    iqae_ci_low: np.ndarray,
    iqae_ci_high: np.ndarray,
    p_true: float,
    output_path: str,
):
    # MC CI half-width: z * sqrt(p(1-p)/N)
    z = 1.96
    mc_n = np.array(sorted({100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200}))
    mc_half = z * np.sqrt(p_true * (1 - p_true) / mc_n)

    iqae_half = 0.5 * (iqae_ci_high - iqae_ci_low)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        mc_n, mc_half,
        lw=2.0,
        label="Classical MC (CI half-width)",
        marker="o", markersize=4, alpha=0.9
    )
    ax.plot(
        iqae_budget, iqae_half,
        lw=2.0,
        label="IQAE (CI half-width)",
        marker="o", markersize=4, alpha=0.9
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(
        ax,
        "CI Half-Width vs Budget",
        "Queries / Samples (log)",
        "CI Half-Width (log)",
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def robustness_plot(output_path: str):
    # Sensitivity: compare two confidence levels for the same distribution
    rng = np.random.default_rng(1234)
    alphas = [0.95, 0.99]
    sample_sizes = np.unique(np.logspace(2, 5, 10, dtype=int))
    theoretical = {
        a: norm.ppf(1 - a, loc=MU, scale=SIGMA) for a in alphas
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha in alphas:
        errors = []
        for n in sample_sizes:
            runs = 20
            run_errs = []
            for _ in range(runs):
                samples = rng.normal(loc=MU, scale=SIGMA, size=int(n))
                var_est = float(np.quantile(samples, 1 - alpha))
                run_errs.append(abs(var_est - theoretical[alpha]))
            errors.append(np.mean(run_errs))
        ax.plot(
            sample_sizes, errors,
            lw=2.0,
            label=f"α = {alpha:.2f}", marker="o", markersize=4, alpha=0.9
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(
        ax,
        "Robustness: Tail Difficulty (Confidence Level)",
        "Samples (log)",
        "VaR Error (log)",
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def mc_convergence_plot(
    mc_stats: Dict[int, Dict[str, float]],
    output_path: str,
):
    sample_sizes = np.array(sorted(mc_stats.keys()))
    means = np.array([mc_stats[n]["var_mean"] for n in sample_sizes])
    p10 = np.array([mc_stats[n]["var_p10"] for n in sample_sizes])
    p90 = np.array([mc_stats[n]["var_p90"] for n in sample_sizes])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sample_sizes, means,
        lw=2.0,
        label="Mean VaR estimate", marker="o", markersize=4, alpha=0.9
    )
    ax.fill_between(
        sample_sizes, p10, p90,
        alpha=0.15, label="10–90 percentile band"
    )

    ax.set_xscale("log")
    style_axes(
        ax,
        "Classical MC Convergence (Variance Bands)",
        "Samples (log)",
        "VaR Estimate",
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    iqae_path = os.path.join(RESULTS_DIR, "iqae_epsilon_sweep.csv")
    classical_path = os.path.join(RESULTS_DIR, "classical_var_convergence.csv")

    if not os.path.exists(iqae_path):
        raise FileNotFoundError(f"Missing IQAE results: {iqae_path}")
    if not os.path.exists(classical_path):
        raise FileNotFoundError(f"Missing classical convergence: {classical_path}")

    iqae_budget, iqae_a_hat, iqae_a_hat_std, iqae_prob_err, iqae_ci_low, iqae_ci_high, a_true = load_iqae_results(iqae_path)
    mc_stats = load_classical_convergence(classical_path)
    if a_true is None or not np.isfinite(a_true):
        raise ValueError("Missing a_true in iqae_epsilon_sweep.csv; re-run value_at_risk.py to regenerate.")

    mc_n = np.array(sorted(mc_stats.keys()))
    mc_err = np.array([mc_stats[n]["mean"] for n in mc_n])

    iqae_var_err = iqae_prob_err
    iqae_var_err_low = np.maximum(iqae_prob_err - iqae_a_hat_std, 0.0)
    iqae_var_err_high = iqae_prob_err + iqae_a_hat_std

    error_vs_budget_plot(
        mc_n, mc_err, iqae_budget, iqae_var_err, iqae_var_err_low, iqae_var_err_high,
        os.path.join(OUTPUT_DIR, "05_error_vs_budget.png"),
    )
    budget_to_target_plot(
        iqae_budget, iqae_ci_low, iqae_ci_high, float(a_true),
        os.path.join(OUTPUT_DIR, "06_budget_to_target.png"),
    )
    ci_halfwidth_vs_budget_plot(
        iqae_budget, iqae_ci_low, iqae_ci_high, float(a_true),
        os.path.join(OUTPUT_DIR, "06b_ci_halfwidth_vs_budget.png"),
    )
    robustness_plot(
        os.path.join(OUTPUT_DIR, "07_robustness_confidence.png"),
    )
    mc_convergence_plot(
        mc_stats,
        os.path.join(OUTPUT_DIR, "08_mc_convergence_bands.png"),
    )

    print("✓ Saved comparison graphs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
