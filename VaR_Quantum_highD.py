

"""
High-Dimensional Quantum VaR via GPU Factor Model + IQAE

Pipeline:
  0. Factor model (GPU)  -- sample Z ~ Student-t(df) in R^d, compute portfolio
                            losses L = -(Z @ B.T @ w) via GPU matrix ops
  1. Discretize          -- histogram L onto a uniform grid of 2^n bins -> PMF
  2. Warm-start bracket  -- use the MC loss samples to get a tight [lo, hi]
                            bracket around the VaR index
  3. State preparation   -- encode PMF as qubit amplitudes
  4. Threshold oracle    -- flip indicator when loss_index < threshold
  5. IQAE                -- estimate P(L <= threshold) with O(1/eps) queries
  6. Bisection           -- binary search within bracket to locate VaR

All high-dimensional complexity lives in step 0 (GPU).
The quantum circuit operates on the 1D marginal loss distribution.

Hybrid story (punchline):
  - GPU makes Step 0 feasible at large d, M, N.
  - Warm start shrinks B (number of IQAE calls) drastically.
  - IQAE improves tail probability estimation per query for tighter epsilon.
"""

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from classiq import *
from classiq.applications.iqae.iqae import IQAE

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NUM_QUBITS = 7          # 2^7 = 128 grid points
ALPHA = 0.05            # VaR confidence: P(L <= VaR) = alpha
TOLERANCE = ALPHA / 10  # bisection convergence tolerance

# IQAE tuning
IQAE_EPSILON = 0.01
IQAE_ALPHA_FAIL = 0.01

# Factor model defaults
D_FACTORS = 5           # factor dimension d
M_ASSETS = 10           # number of assets M
STUDENT_T_DF = 4        # heavy-tail degrees of freedom (no skew)
N_MC = 100_000          # GPU MC sample count
RHO = 0.3               # factor correlation (equicorrelated)
N_SECTORS = 5
MARKET_STRENGTH = 0.7
SECTOR_STRENGTH = 1.0
NOISE_STRENGTH = 0.1


# ---------------------------------------------------------------------------
# Factor model
# ---------------------------------------------------------------------------
@dataclass
class FactorModel:
    d: int              # factor dimension
    M: int              # number of assets
    df: float           # Student-t degrees of freedom
    B: np.ndarray       # (M, d) factor loadings
    w: np.ndarray       # (M,) portfolio weights

    @staticmethod
    def random(d, M, df, seed=0):
        """Random loadings B ~ N(0, 1/sqrt(d)), equal weights w = 1/M."""
        rng = np.random.default_rng(seed)
        B = rng.standard_normal((M, d)) / np.sqrt(d)
        w = np.full(M, 1.0 / M)
        return FactorModel(d=d, M=M, df=df, B=B, w=w)

    @staticmethod
    def structured(
        d,
        M,
        df,
        n_sectors=N_SECTORS,
        seed=0,
        market_strength=MARKET_STRENGTH,
        sector_strength=SECTOR_STRENGTH,
        noise=NOISE_STRENGTH,
    ):
        """
        Structured loadings:
          - factor 0 = market (loads on all assets)
          - next factors = sector-specific loadings
          - small idiosyncratic noise
        """
        rng = np.random.default_rng(seed)
        B = noise * rng.standard_normal((M, d)) / np.sqrt(d)

        # factor 0 = market
        if d >= 1:
            B[:, 0] += market_strength

        # next factors are sector factors
        if d >= 2:
            sectors = np.array_split(np.arange(M), n_sectors)
            for s, idxs in enumerate(sectors):
                f = 1 + (s % (d - 1))  # pick a factor index 1..d-1
                B[idxs, f] += sector_strength

        w = np.full(M, 1.0 / M)
        return FactorModel(d=d, M=M, df=df, B=B, w=w)

def _equicorrelated_normal(key_or_rng, n_samples, d, rho, backend="jax"):
    """
    Sample G ~ N(0, Sigma) where Sigma = (1-rho)*I + rho*11^T.

    Uses the rank-1 trick:  G_i = sqrt(rho)*z_common + sqrt(1-rho)*z_indep_i
    Cost: O(N*d) memory and compute.  No d×d matrix needed.
    """
    a = np.sqrt(rho)
    b = np.sqrt(1.0 - rho)

    if backend == "jax":
        import jax
        import jax.numpy as jnp
        k1, k2 = jax.random.split(key_or_rng)
        z_common = jax.random.normal(k1, (n_samples, 1))       # (N, 1)
        z_indep = jax.random.normal(k2, (n_samples, d))        # (N, d)
        return a * z_common + b * z_indep                       # (N, d)
    else:
        z_common = key_or_rng.standard_normal((n_samples, 1))
        z_indep = key_or_rng.standard_normal((n_samples, d))
        return a * z_common + b * z_indep


def gpu_sample_losses(model, n_samples, alpha=ALPHA, seed=0):
    """
    GPU-accelerated Monte Carlo: sample factors, compute portfolio losses.

    Z ~ multivariate Student-t(df, 0, Sigma) with equicorrelated Sigma,
    sampled via the rank-1 trick (no d×d matrix, scales to d=100k+).

    r = Z @ B.T           -> (N, M)  asset returns
    P = r @ w             -> (N,)    portfolio P&L
    L = -P                -> (N,)    losses

    Backend priority: JAX (GPU/TPU) > numpy (CPU fallback).
    """
    device = "cpu"
    rho = RHO

    try:
        import jax
        import jax.numpy as jnp

        # JAX device selection
        backend = jax.default_backend()          # 'gpu', 'tpu', or 'cpu'
        device = backend if backend != "cpu" else "jax-cpu"

        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)

        # Equicorrelated Gaussian factors  (N, d) — O(N*d), no Cholesky
        G = _equicorrelated_normal(k1, n_samples, model.d, rho, backend="jax")

        # Student-t: Z = G * sqrt(df / chi2(df))
        # chi2(df) = 2 * Gamma(df/2, 1) — gamma is available in all JAX versions
        chi2_samples = 2.0 * jax.random.gamma(k2, model.df / 2.0, shape=(n_samples, 1))
        Z = G * jnp.sqrt(model.df / chi2_samples)          # (N, d)

        # Portfolio losses
        B_j = jnp.array(model.B, dtype=jnp.float32)        # (M, d)
        w_j = jnp.array(model.w, dtype=jnp.float32)        # (M,)
        r = Z @ B_j.T                                       # (N, M)
        L = -(r @ w_j)                                      # (N,)

        loss_np = np.asarray(L, dtype=np.float64)

    except ImportError:
        # Pure numpy fallback — same rank-1 trick, no d×d matrix
        device = "numpy-cpu"
        rng = np.random.default_rng(seed)

        G = _equicorrelated_normal(rng, n_samples, model.d, rho, backend="numpy")

        U = rng.chisquare(model.df, size=(n_samples, 1))
        Z = G * np.sqrt(model.df / U)

        r = Z @ model.B.T
        P = r @ model.w
        loss_np = (-P).astype(np.float64)

    var_mc = float(np.percentile(loss_np, alpha * 100))

    return {
        "loss_samples": loss_np,
        "var_mc": var_mc,
        "n_samples": n_samples,
        "device": device,
    }


# ---------------------------------------------------------------------------
# 1D Student-t GPU sampling (for warm start without factor model)
# ---------------------------------------------------------------------------
def gpu_sample_1d_student_t(df, n_samples, alpha=ALPHA, seed=0):
    """
    Sample from 1D Student-t(df) on GPU (JAX) or CPU (numpy).

    Used for warm-start bracket estimation in the 1D benchmark path.
    """
    device = "cpu"
    try:
        import jax
        import jax.numpy as jnp

        backend = jax.default_backend()
        device = backend if backend != "cpu" else "jax-cpu"

        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)

        # Student-t = normal / sqrt(chi2/df)
        G = jax.random.normal(k1, (n_samples,))
        chi2 = 2.0 * jax.random.gamma(k2, df / 2.0, shape=(n_samples,))
        samples = G * jnp.sqrt(df / chi2)

        loss_np = np.asarray(samples, dtype=np.float64)

    except ImportError:
        device = "numpy-cpu"
        rng = np.random.default_rng(seed)
        loss_np = rng.standard_t(df, size=n_samples).astype(np.float64)

    return {
        "loss_samples": loss_np,
        "var_mc": float(np.percentile(loss_np, alpha * 100)),
        "n_samples": n_samples,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Analytic 1D Student-t PMF — uniform discretization
# ---------------------------------------------------------------------------
def build_analytic_student_t_pmf(df, num_qubits, n_sigmas=4):
    """
    Discretize a standard Student-t(df) onto 2^num_qubits uniform bins.

    Grid spans [-n_sigmas*scale, +n_sigmas*scale] where scale = sqrt(df/(df-2))
    for df > 2 (the std dev of the t distribution).

    Returns (grid_centers, probs_list, lo, hi).
    """
    dist = scipy.stats.t(df=df)
    scale = np.sqrt(df / (df - 2)) if df > 2 else 3.0
    lo = -n_sigmas * scale
    hi = n_sigmas * scale

    n_bins = 2 ** num_qubits
    edges = np.linspace(lo, hi, n_bins + 1)

    # Exact bin probabilities from CDF differences
    cdf_vals = dist.cdf(edges)
    pmf = np.diff(cdf_vals)

    # Classiq needs strictly positive probs
    pmf = np.maximum(pmf, 1e-10)
    pmf = pmf / pmf.sum()

    grid = 0.5 * (edges[:-1] + edges[1:])
    return grid, pmf.tolist(), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Analytic 1D Student-t PMF — tail-aware discretization
# ---------------------------------------------------------------------------
def build_tail_aware_pmf(df, num_qubits, alpha=ALPHA,
                         lo_quantile=0.001, hi_quantile=0.30):
    """
    Concentrate all 2^num_qubits bins around the left tail near the VaR.

    Grid spans [ppf(lo_quantile), ppf(hi_quantile)] — a narrow window
    that captures the VaR region with much finer resolution than the
    full-range uniform grid.

    Probability mass outside the grid is folded into the edge bins so
    the PMF still sums to 1.

    For alpha=0.05, hi_quantile=0.30: the VaR falls around index
    ~(0.05/0.30)*128 ≈ 21, giving ~6x finer resolution than uniform.

    Returns (grid_centers, probs_list, lo, hi).
    """
    dist = scipy.stats.t(df=df)
    lo = float(dist.ppf(lo_quantile))
    hi = float(dist.ppf(hi_quantile))

    n_bins = 2 ** num_qubits
    edges = np.linspace(lo, hi, n_bins + 1)

    # Exact bin probabilities from CDF
    cdf_vals = dist.cdf(edges)
    pmf = np.diff(cdf_vals)

    # Fold tail mass outside the grid into edge bins
    pmf[0] += cdf_vals[0]              # mass below grid  (= lo_quantile)
    pmf[-1] += 1.0 - cdf_vals[-1]      # mass above grid  (= 1 - hi_quantile)

    # Classiq needs strictly positive probs
    pmf = np.maximum(pmf, 1e-10)
    pmf = pmf / pmf.sum()

    grid = 0.5 * (edges[:-1] + edges[1:])
    return grid, pmf.tolist(), lo, hi


# ---------------------------------------------------------------------------
# Discretize losses to PMF
# ---------------------------------------------------------------------------
def build_pmf_from_samples(loss_samples, num_qubits, lo_pct=0.1, hi_pct=99.9):
    """
    Histogram loss samples onto a uniform grid of 2^num_qubits bins.

    Returns (grid_centers, probs_list, lo, hi).
    """
    lo = float(np.percentile(loss_samples, lo_pct))
    hi = float(np.percentile(loss_samples, hi_pct))
    # Ensure lo < hi
    if hi <= lo:
        hi = lo + 1.0
    n_bins = 2 ** num_qubits

    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(loss_samples, bins=edges)

    total = counts.sum()
    if total == 0:
        pmf = np.full(n_bins, 1.0 / n_bins)
    else:
        pmf = counts.astype(float) / total

    # Classiq needs strictly positive probs for state prep
    pmf = np.maximum(pmf, 1e-10)
    pmf = pmf / pmf.sum()

    grid = 0.5 * (edges[:-1] + edges[1:])
    return grid, pmf.tolist(), lo, hi


# ---------------------------------------------------------------------------
# Module-level placeholder (overridden in __main__ with factor model data)
# ---------------------------------------------------------------------------
grid_points = np.linspace(-1, 1, 2 ** NUM_QUBITS)
probs = [1.0 / (2 ** NUM_QUBITS)] * (2 ** NUM_QUBITS)
GRID_LO, GRID_HI = -1.0, 1.0

# Mutable threshold index consumed by the oracle
THRESHOLD_INDEX: int = 0


# ---------------------------------------------------------------------------
# Classical VaR (reference baseline)
# ---------------------------------------------------------------------------
def classical_var(grid, pmf, alpha):
    """Accumulate probabilities to find the first index where CDF >= alpha."""
    cumulative = 0.0
    for i, p in enumerate(pmf):
        cumulative += p
        if cumulative >= alpha:
            return i, float(grid[i])
    return len(grid) - 1, float(grid[-1])


# ---------------------------------------------------------------------------
# Warm start from raw loss samples
# ---------------------------------------------------------------------------
def mc_warm_start_from_losses(loss_samples, grid, alpha, confidence=0.99):
    """
    Compute a warm-start bracket directly from GPU MC loss samples.

    No re-sampling needed -- uses the raw loss vector from gpu_sample_losses.
    Maps quantile CI in loss-space back to grid indices via searchsorted.
    """
    n = len(loss_samples)
    sorted_losses = np.sort(loss_samples)

    # Alpha-quantile point estimate
    k_center = max(0, min(int(np.floor(alpha * n)), n - 1))
    var_est_loss = float(sorted_losses[k_center])

    # Binomial order-statistic CI
    beta = 1.0 - confidence
    j_lo = max(0, int(scipy.stats.binom.ppf(beta / 2, n, alpha)))
    j_hi = min(n - 1, int(scipy.stats.binom.ppf(1 - beta / 2, n, alpha)))

    loss_lo = float(sorted_losses[j_lo])
    loss_hi = float(sorted_losses[j_hi])

    # Map loss values to grid indices (grid is sorted ascending)
    var_est_idx = int(np.clip(np.searchsorted(grid, var_est_loss), 0, len(grid) - 1))
    lo_idx = int(np.clip(np.searchsorted(grid, loss_lo) - 1, 0, len(grid) - 1))
    hi_idx = int(np.clip(np.searchsorted(grid, loss_hi) + 1, 0, len(grid) - 1))

    return {
        "lo": lo_idx,
        "hi": hi_idx,
        "var_est_idx": var_est_idx,
        "var_est_value": float(grid[var_est_idx]),
        "n_samples": n,
        "method": "factor",
        "bracket_width": hi_idx - lo_idx,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_distribution(
    grid,
    pmf,
    loss_samples=None,
    var_val=None,
    model=None,
    output_path="results/highd_distribution.png",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    if loss_samples is not None:
        ax.hist(
            loss_samples,
            bins=200,
            density=True,
            alpha=0.3,
            color="steelblue",
            label="MC loss histogram",
        )

    dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
    ax.bar(
        grid,
        np.array(pmf) / dx,
        width=dx * 0.9,
        alpha=0.6,
        color="orange",
        label="Discretized PMF",
    )

    if var_val is not None:
        ax.axvline(
            x=var_val,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"VaR ({ALPHA * 100:.0f}%)",
        )

    title = "Portfolio Loss Distribution"
    if model is not None:
        title += f" (d={model.d}, M={model.M}, df={model.df})"
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution plot to {output_path}")


# ---------------------------------------------------------------------------
# Quantum circuit (Classiq) -- unchanged
# ---------------------------------------------------------------------------
@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    """Full A operator: load distribution then apply threshold oracle."""
    load_distribution(asset=asset)
    payoff(asset=asset, ind=ind)


@qfunc
def load_distribution(asset: QNum):
    """Encode the loss PMF into qubit amplitudes."""
    inplace_prepare_state(probs, bound=0, target=asset)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    """Threshold oracle: flip |ind> when loss_index < THRESHOLD_INDEX."""
    ind ^= asset < THRESHOLD_INDEX


# ---------------------------------------------------------------------------
# IQAE helpers -- unchanged
# ---------------------------------------------------------------------------
def run_iqae_detailed(iqae, epsilon, alpha_fail):
    """Run a single IQAE call and return estimation + resource metrics."""
    result = iqae.run(epsilon=epsilon, alpha=alpha_fail)

    estimation = float(result.estimation)
    ci_low = float(result.confidence_interval[0])
    ci_high = float(result.confidence_interval[1])

    iterations_data = getattr(result, "iterations_data", []) or []
    shots_total = 0
    grover_calls = 0
    ks_used = []
    for it in iterations_data:
        k = getattr(it, "grover_iterations", None)
        ks_used.append(k)
        shots = None
        if hasattr(it, "sample_results") and it.sample_results is not None:
            shots = getattr(it.sample_results, "num_shots", None)
        if hasattr(it, "num_shots") and shots is None:
            shots = it.num_shots
        shots_total += shots or 0
        if k is not None and shots is not None:
            grover_calls += k * shots

    if shots_total == 0 and hasattr(result, "sample_results"):
        shots_total = getattr(result.sample_results, "num_shots", 0)

    return {
        "estimation": estimation,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "shots_total": shots_total,
        "grover_calls": grover_calls,
        "ks_used": ks_used,
    }


# ---------------------------------------------------------------------------
# IQAE tail-probability estimation -- unchanged
# ---------------------------------------------------------------------------
def estimate_tail_probability(threshold_index, epsilon=IQAE_EPSILON,
                              alpha_fail=IQAE_ALPHA_FAIL):
    """Use IQAE to estimate P(loss_index < threshold_index)."""
    global THRESHOLD_INDEX
    THRESHOLD_INDEX = int(threshold_index)

    iqae = IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=NUM_QUBITS,
        constraints=Constraints(max_width=28),
        preferences=Preferences(machine_precision=NUM_QUBITS),
    )

    res = run_iqae_detailed(iqae, epsilon=epsilon, alpha_fail=alpha_fail)
    ci = (res["ci_low"], res["ci_high"])
    return res["estimation"], ci, res


# ---------------------------------------------------------------------------
# Quantum VaR via bisection -- unchanged
# ---------------------------------------------------------------------------
def quantum_value_at_risk(grid, pmf, alpha, tolerance=TOLERANCE,
                          epsilon=IQAE_EPSILON, alpha_fail=IQAE_ALPHA_FAIL,
                          bracket=None):
    """
    IQAE + bisection. With bracket: log2(hi-lo) steps. Without: log2(n).
    """
    n = len(grid)
    total_oracle_queries = 0
    bisection_steps = 0
    est, ci = None, None

    if bracket is not None:
        lo, hi = bracket
        while hi - lo > 1:
            mid = (lo + hi) // 2
            est, ci, res = estimate_tail_probability(mid, epsilon, alpha_fail)
            total_oracle_queries += res["grover_calls"]
            bisection_steps += 1
            print(f"  [{lo},{hi}] mid={mid}  P_hat={est:.6f}"
                  f"  ci=[{ci[0]:.6f}, {ci[1]:.6f}]"
                  f"  shots={res['shots_total']}  grover={res['grover_calls']}")
            if est < alpha:
                lo = mid
            else:
                hi = mid

        if est is None:
            est, ci, res = estimate_tail_probability(hi, epsilon, alpha_fail)
            total_oracle_queries += res["grover_calls"]
            bisection_steps += 1
            print(f"  confirm idx={hi}  P_hat={est:.6f}"
                  f"  ci=[{ci[0]:.6f}, {ci[1]:.6f}]"
                  f"  shots={res['shots_total']}  grover={res['grover_calls']}")

        return hi, float(grid[hi]), est, ci, total_oracle_queries, bisection_steps

    index = n // 4
    search_size = index // 2

    est, ci, res = estimate_tail_probability(index, epsilon, alpha_fail)
    total_oracle_queries += res["grover_calls"]
    bisection_steps += 1
    print(f"  idx={index:<4d}  P_hat={est:.6f}  ci=[{ci[0]:.6f}, {ci[1]:.6f}]"
          f"  shots={res['shots_total']}  grover={res['grover_calls']}")

    while not np.isclose(est, alpha, atol=tolerance) and search_size > 0:
        if est < alpha:
            index += search_size
        else:
            index -= search_size
        search_size //= 2
        est, ci, res = estimate_tail_probability(index, epsilon, alpha_fail)
        total_oracle_queries += res["grover_calls"]
        bisection_steps += 1
        print(f"  idx={index:<4d}  P_hat={est:.6f}  ci=[{ci[0]:.6f}, {ci[1]:.6f}]"
              f"  shots={res['shots_total']}  grover={res['grover_calls']}")

    return index, float(grid[index]), est, ci, total_oracle_queries, bisection_steps


# ---------------------------------------------------------------------------
# IQAE epsilon sweep -- unchanged
# ---------------------------------------------------------------------------
def iqae_epsilon_sweep(epsilons, alpha_fail=IQAE_ALPHA_FAIL,
                       output_path="results/iqae_epsilon_sweep.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ref_idx, _ = classical_var(grid_points, probs, ALPHA)
    global THRESHOLD_INDEX
    THRESHOLD_INDEX = int(ref_idx)
    a_true = sum(probs[:THRESHOLD_INDEX])

    iqae = IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=NUM_QUBITS,
        constraints=Constraints(max_width=28),
        preferences=Preferences(machine_precision=NUM_QUBITS),
    )

    fieldnames = [
        "epsilon", "alpha_fail", "a_true", "a_hat",
        "abs_error", "ci_low", "ci_high",
        "shots_total", "grover_calls", "ks_used",
    ]

    if isinstance(alpha_fail, (list, tuple)):
        alpha_fails = list(alpha_fail)
    else:
        alpha_fails = sorted({alpha_fail, 0.1, 0.05, 0.02, 0.01, 0.005})

    rows = []
    for a_fail in alpha_fails:
        for eps in epsilons:
            res = run_iqae_detailed(iqae, epsilon=eps, alpha_fail=a_fail)
            abs_error = abs(res["estimation"] - a_true)
            row = {
                "epsilon": eps, "alpha_fail": a_fail,
                "a_true": a_true, "a_hat": res["estimation"],
                "abs_error": abs_error,
                "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "shots_total": res["shots_total"],
                "grover_calls": res["grover_calls"],
                "ks_used": ";".join(str(k) for k in res["ks_used"] if k is not None),
            }
            rows.append(row)
            print(
                f"  alpha_fail={a_fail:>6.3f}  eps={eps:>5.3f}  "
                f"a_hat={res['estimation']:.6f}  "
                f"abs_err={abs_error:.6f}  shots={res['shots_total']}  "
                f"grover={res['grover_calls']}"
            )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSweep results written to {output_path}")
    return rows


# ---------------------------------------------------------------------------
# Benchmark: GPU warm start + IQAE sweep over epsilon
# ---------------------------------------------------------------------------
def benchmark_epsilon_sweep(grid, pmf, loss_samples, epsilons, alpha_fails,
                            alpha=ALPHA,
                            mc_confidence=0.99,
                            output_path="results/benchmark_sweep.csv"):
    """
    Full-pipeline benchmark: for each (epsilon, alpha_fail) pair, run
    GPU warm start → IQAE bisection and record VaR estimate, error, and
    oracle cost.

    The warm-start bracket is computed once (it doesn't depend on epsilon).
    Nested loop: outer = epsilon, inner = alpha_fail (matches quantum_iqae.py).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Classical reference (ground truth on the discretized grid)
    ref_idx, ref_val = classical_var(grid, pmf, alpha)

    # Analytic reference (if distribution is standard Student-t)
    analytic_var = None
    try:
        analytic_var = float(scipy.stats.t.ppf(alpha, df=_BENCH_DF))
    except Exception:
        pass

    # GPU warm-start bracket (computed once)
    ws = mc_warm_start_from_losses(loss_samples, grid, alpha,
                                   confidence=mc_confidence)
    bracket = (ws["lo"], ws["hi"])

    print(f"  Reference: classical CDF index={ref_idx}  value={ref_val:.6f}")
    if analytic_var is not None:
        print(f"  Reference: analytic ppf={analytic_var:.6f}")
    print(f"  Warm-start bracket: [{bracket[0]}, {bracket[1]}]  "
          f"(width={ws['bracket_width']})")
    print(f"  Sweeping {len(epsilons)} epsilons x {len(alpha_fails)} alpha_fails "
          f"= {len(epsilons) * len(alpha_fails)} runs")
    print()

    fieldnames = [
        "epsilon", "alpha_fail",
        "est_index", "est_var",
        "ref_index", "ref_var",
        "abs_error", "abs_error_vs_analytic",
        "oracle_queries_total", "bisection_steps",
        "bracket_width", "n_mc_samples",
    ]

    results = []
    for eps in epsilons:
        for af in alpha_fails:
            print(f"  --- eps={eps:.4f}  alpha_fail={af:.4f} ---")
            q_idx, q_var, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk(
                grid, pmf, alpha,
                epsilon=eps, alpha_fail=af,
                bracket=bracket,
            )

            err = abs(q_var - ref_val)
            err_analytic = abs(q_var - analytic_var) if analytic_var is not None else None

            row = {
                "epsilon": eps,
                "alpha_fail": af,
                "est_index": q_idx,
                "est_var": q_var,
                "ref_index": ref_idx,
                "ref_var": ref_val,
                "abs_error": err,
                "abs_error_vs_analytic": err_analytic,
                "oracle_queries_total": q_queries,
                "bisection_steps": q_steps,
                "bracket_width": ws["bracket_width"],
                "n_mc_samples": ws["n_samples"],
            }
            results.append(row)

            print(f"  eps={eps:.4f}  af={af:.4f}  q_idx={q_idx}  q_var={q_var:.6f}  "
                  f"err={err:.6f}  queries={q_queries}  steps={q_steps}")
            print()

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark results written to {output_path}")

    return results


# Module-level variable set before benchmark runs (used by benchmark_epsilon_sweep)
_BENCH_DF = STUDENT_T_DF


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="High-D Quantum VaR: GPU factor model + IQAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Factor model (default)
  python VaR_Quantum_highD.py --d-factors 20 --m-assets 50

  # 1D benchmark: GPU warm start + IQAE, sweep epsilon, uniform bins
  python VaR_Quantum_highD.py --benchmark --df 4 --discretization uniform

  # 1D benchmark: tail-aware bins for better VaR resolution
  python VaR_Quantum_highD.py --benchmark --df 4 --discretization tail

  # Pure IQAE (no MC, no warm start)
  python VaR_Quantum_highD.py --no-mc --df 4

  # Factor model with sweep
  python VaR_Quantum_highD.py --d-factors 1000 --m-assets 100 --structured --sweep
""",
    )
    # Mode
    p.add_argument("--no-mc", action="store_true",
                   help="skip factor model + GPU MC; use analytic Student-t(df) PMF "
                        "for pure IQAE benchmarking")
    p.add_argument("--benchmark", action="store_true",
                   help="1D benchmark: GPU sample Student-t → discretize → "
                        "warm start → IQAE sweep over epsilon")
    p.add_argument("--discretization", choices=["uniform", "tail"],
                   default="uniform",
                   help="PMF discretization method (default: uniform)")
    p.add_argument("--tail-lo-q", type=float, default=0.001,
                   help="tail-aware: lower quantile for grid range (default: 0.001)")
    p.add_argument("--tail-hi-q", type=float, default=0.30,
                   help="tail-aware: upper quantile for grid range (default: 0.30)")
    # Factor model
    p.add_argument("--d-factors", type=int, default=D_FACTORS,
                   help=f"factor dimension d (default: {D_FACTORS})")
    p.add_argument("--m-assets", type=int, default=M_ASSETS,
                   help=f"number of assets M (default: {M_ASSETS})")
    p.add_argument("--df", type=int, default=STUDENT_T_DF,
                   help=f"Student-t degrees of freedom (default: {STUDENT_T_DF})")
    p.add_argument("--n-mc", type=int, default=N_MC,
                   help=f"GPU MC samples (default: {N_MC})")
    p.add_argument("--rho", type=float, default=RHO,
                   help=f"equicorrelation for factors (default: {RHO})")
    p.add_argument("--structured", action="store_true",
                   help="use structured loadings (market + sector factors)")
    p.add_argument("--n-sectors", type=int, default=N_SECTORS,
                   help=f"number of sectors (default: {N_SECTORS})")
    p.add_argument("--market-strength", type=float, default=MARKET_STRENGTH,
                   help=f"market factor strength (default: {MARKET_STRENGTH})")
    p.add_argument("--sector-strength", type=float, default=SECTOR_STRENGTH,
                   help=f"sector factor strength (default: {SECTOR_STRENGTH})")
    p.add_argument("--noise", type=float, default=NOISE_STRENGTH,
                   help=f"idiosyncratic noise scale (default: {NOISE_STRENGTH})")
    # Warm start
    p.add_argument("--no-warmstart", action="store_true",
                   help="skip MC warm start, run full IQAE bisection over all bins")
    p.add_argument("--skip-threshold", type=int, default=1,
                   help="skip IQAE if bracket width <= this (default: 1)")
    p.add_argument("--mc-confidence", type=float, default=0.99,
                   help="CI confidence for bracket (default: 0.99)")
    # IQAE
    p.add_argument("--epsilon", type=float, default=IQAE_EPSILON,
                   help=f"IQAE precision target for single-run mode (default: {IQAE_EPSILON})")
    p.add_argument("--alpha-fail", type=float, default=IQAE_ALPHA_FAIL,
                   help=f"IQAE failure probability for single-run mode (default: {IQAE_ALPHA_FAIL})")
    # Benchmark sweep lists (used with --benchmark)
    p.add_argument("--epsilon-list", type=float, nargs="+",
                   default=[0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05,
                            0.04, 0.03, 0.025, 0.02, 0.015, 0.01],
                   help="epsilon values to sweep in --benchmark mode "
                        "(default: 13 values from 0.20 to 0.01)")
    p.add_argument("--alpha-fail-list", type=float, nargs="+",
                   default=[0.01],
                   help="alpha_fail values to sweep in --benchmark mode "
                        "(default: [0.01])")
    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-dist", action="store_true",
                   help="save distribution plot")
    p.add_argument("--sweep", action="store_true",
                   help="run IQAE epsilon sweep (fixed-threshold, no bisection)")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = build_parser().parse_args()

    # =================================================================
    # Branch: --benchmark | --no-mc | factor model (default)
    # =================================================================
    mc = None
    model = None
    warmup = None

    if args.benchmark:
        # ----- 1D Benchmark: GPU sample → discretize → warm start → IQAE sweep -----
        _BENCH_DF = args.df

        disc_label = args.discretization
        print(f"Mode: 1D benchmark  Student-t(df={args.df})  "
              f"discretization={disc_label}")
        print()

        # Step A: GPU sample from 1D Student-t
        print(f"GPU sampling {args.n_mc:,} Student-t(df={args.df}) values ...")
        mc = gpu_sample_1d_student_t(args.df, args.n_mc, alpha=ALPHA, seed=args.seed)
        print(f"  device      : {mc['device']}")
        print(f"  sample range: [{mc['loss_samples'].min():.4f}, "
              f"{mc['loss_samples'].max():.4f}]")
        print(f"  MC VaR est  : {mc['var_mc']:.6f}")
        print()

        # Step B: Discretize
        if disc_label == "tail":
            print(f"Tail-aware discretization: quantiles "
                  f"[{args.tail_lo_q}, {args.tail_hi_q}]")
            grid_points, probs, GRID_LO, GRID_HI = build_tail_aware_pmf(
                args.df, NUM_QUBITS, alpha=ALPHA,
                lo_quantile=args.tail_lo_q, hi_quantile=args.tail_hi_q,
            )
        else:
            grid_points, probs, GRID_LO, GRID_HI = build_analytic_student_t_pmf(
                args.df, NUM_QUBITS,
            )

        bin_width = (GRID_HI - GRID_LO) / (2 ** NUM_QUBITS)
        print(f"Discretized onto {2**NUM_QUBITS} bins over "
              f"[{GRID_LO:.4f}, {GRID_HI:.4f}]  bin_width={bin_width:.6f}")

        # References
        analytic_var = float(scipy.stats.t.ppf(ALPHA, df=args.df))
        ref_idx, ref_var = classical_var(grid_points, probs, ALPHA)
        print(f"Analytic VaR (ppf)       : {analytic_var:.6f}")
        print(f"Classical CDF VaR (grid) : {ref_var:.6f}  (index {ref_idx})")
        print(f"MC VaR (raw percentile)  : {mc['var_mc']:.6f}")
        print()

        if args.plot_dist:
            plot_distribution(
                grid_points, probs,
                loss_samples=mc["loss_samples"],
                var_val=ref_var,
                output_path=f"results/benchmark_{disc_label}_df{args.df}.png",
            )

        # Step C: Sweep (epsilon x alpha_fail) with warm start
        epsilons = args.epsilon_list
        alpha_fails = args.alpha_fail_list

        print(f"Sweeping {len(epsilons)} epsilons x {len(alpha_fails)} alpha_fails")
        print(f"  epsilon    : {epsilons}")
        print(f"  alpha_fail : {alpha_fails}")
        print()

        results = benchmark_epsilon_sweep(
            grid_points, probs, mc["loss_samples"],
            epsilons, alpha_fails,
            alpha=ALPHA,
            mc_confidence=args.mc_confidence,
            output_path=f"results/benchmark_{disc_label}_df{args.df}.csv",
        )

        # Summary table
        print()
        print("=" * 90)
        print(f"BENCHMARK SUMMARY  (df={args.df}, {disc_label}, "
              f"{2**NUM_QUBITS} bins, {mc['n_samples']:,} MC samples)")
        print("=" * 90)
        print(f"{'eps':>7s}  {'af':>6s}  {'q_var':>10s}  {'ref_var':>10s}  "
              f"{'err':>10s}  {'queries':>8s}  {'steps':>5s}  {'bracket':>7s}")
        print("-" * 90)
        for r in results:
            print(f"{r['epsilon']:7.4f}  {r['alpha_fail']:6.4f}  "
                  f"{r['est_var']:10.6f}  {r['ref_var']:10.6f}  "
                  f"{r['abs_error']:10.6f}  {r['oracle_queries_total']:8d}  "
                  f"{r['bisection_steps']:5d}  {r['bracket_width']:7d}")
        print("=" * 90)

    elif args.no_mc:
        # ----- Analytic Student-t(df) path: no sampling at all -----
        print(f"Mode: analytic Student-t(df={args.df}) — no MC sampling")
        print()

        grid_points, probs, GRID_LO, GRID_HI = build_analytic_student_t_pmf(
            args.df, NUM_QUBITS,
        )
        print(f"Discretized onto {2**NUM_QUBITS} bins over [{GRID_LO:.4f}, {GRID_HI:.4f}]")

        # Exact VaR from the analytic CDF
        analytic_var = float(scipy.stats.t.ppf(ALPHA, df=args.df))
        print(f"Analytic VaR (Student-t ppf) : {analytic_var:.6f}")

        # Classical CDF on the discretized PMF
        ref_idx, ref_var = classical_var(grid_points, probs, ALPHA)
        print(f"Classical CDF VaR ({2**NUM_QUBITS} bins) : {ref_var:.6f}  (index {ref_idx})")
        print()

        if args.plot_dist:
            plot_distribution(grid_points, probs, var_val=ref_var,
                              output_path="results/analytic_t_distribution.png")

        # Pure IQAE bisection — no warm start
        print(f"Running quantum VaR (IQAE + full bisection over {2**NUM_QUBITS} bins) ...")
        q_idx, q_var, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk(
            grid_points, probs, ALPHA,
            epsilon=args.epsilon,
            alpha_fail=args.alpha_fail,
            bracket=None,
        )

        # --- Results ---
        print()
        print("=" * 70)
        print("RESULTS COMPARISON  (analytic mode, no MC)")
        print("=" * 70)

        print()
        print(f"1) Analytic Student-t(df={args.df}) VaR")
        print(f"   VaR at {ALPHA*100:.0f}%          : {analytic_var:.6f}")

        print()
        print(f"2) Classical CDF on discretized PMF ({2**NUM_QUBITS} bins)")
        print(f"   VaR at {ALPHA*100:.0f}%          : {ref_var:.6f}")
        print(f"   Grid index          : {ref_idx}")
        print(f"   Discretization      : [{GRID_LO:.4f}, {GRID_HI:.4f}]")

        print()
        print(f"3) Quantum IQAE + full bisection")
        print(f"   VaR at {ALPHA*100:.0f}%          : {q_var:.6f}")
        print(f"   Grid index          : {q_idx}")
        if q_est is not None:
            print(f"   Tail P estimate     : {q_est:.6f}")
            print(f"   IQAE CI             : [{q_ci[0]:.6f}, {q_ci[1]:.6f}]")
        print(f"   Bisection steps     : {q_steps}")
        print(f"   Oracle queries      : {q_queries}")

        print()
        print("Deltas:")
        print(f"   Analytic vs Discretized  : {abs(analytic_var - ref_var):.6f}")
        print(f"   Analytic vs Quantum      : {abs(analytic_var - q_var):.6f}")
        print(f"   Discretized vs Quantum   : {abs(ref_var - q_var):.6f}")

        print()
        print("-" * 70)
        print("COST BREAKDOWN  (pure IQAE, no MC)")
        print("-" * 70)
        print(f"  MC samples            : 0  (analytic PMF)")
        print(f"  Bisection steps       : {q_steps}")
        print(f"  Oracle queries        : {q_queries}")
        print(f"  IQAE convergence      : O(1/eps)     eps={args.epsilon}")
        print("=" * 70)

    else:
        # ----- Factor model + GPU MC path -----

        # --- Step 0: Build factor model ---
        if args.structured:
            model = FactorModel.structured(
                args.d_factors,
                args.m_assets,
                args.df,
                n_sectors=args.n_sectors,
                seed=args.seed,
                market_strength=args.market_strength,
                sector_strength=args.sector_strength,
                noise=args.noise,
            )
        else:
            model = FactorModel.random(args.d_factors, args.m_assets, args.df, seed=args.seed)
        print(f"Factor model: d={model.d}, M={model.M}, df={model.df}")
        print(f"  B shape: {model.B.shape}  w shape: {model.w.shape}")
        print()

        # --- Step 1: GPU Monte Carlo ---
        print(f"GPU MC sampling ({args.n_mc:,} scenarios) ...")
        mc = gpu_sample_losses(model, args.n_mc, alpha=ALPHA, seed=args.seed)
        print(f"  device      : {mc['device']}")
        print(f"  loss range  : [{mc['loss_samples'].min():.4f}, {mc['loss_samples'].max():.4f}]")
        print(f"  MC VaR est  : {mc['var_mc']:.6f}")
        print()

        # --- Step 2: Discretize losses to PMF ---
        grid_points, probs, GRID_LO, GRID_HI = build_pmf_from_samples(
            mc["loss_samples"], NUM_QUBITS,
        )
        print(f"Discretized onto {2**NUM_QUBITS} bins over [{GRID_LO:.4f}, {GRID_HI:.4f}]")

        # --- Classical reference ---
        ref_idx, ref_var = classical_var(grid_points, probs, ALPHA)
        print(f"Classical VaR at {ALPHA*100:.0f}%: {ref_var:.6f}  (index {ref_idx})")
        print()

        # --- Plot ---
        if args.plot_dist:
            plot_distribution(
                grid_points,
                probs,
                loss_samples=mc["loss_samples"],
                var_val=ref_var,
                model=model,
            )

        # --- Step 3: Warm start bracket from loss samples ---
        if not args.no_warmstart:
            warmup = mc_warm_start_from_losses(
                mc["loss_samples"], grid_points, ALPHA,
                confidence=args.mc_confidence,
            )
            print(f"Warm start bracket: [{warmup['lo']}, {warmup['hi']}]  "
                  f"(width={warmup['bracket_width']})")
            print(f"  MC VaR index: {warmup['var_est_idx']}  "
                  f"value: {warmup['var_est_value']:.6f}")
            print()
        else:
            print("Warm start disabled (--no-warmstart). Full IQAE bisection.")
            print()

        # --- Step 4: Quantum VaR ---
        skipped_iqae = False
        if warmup is not None and warmup["bracket_width"] <= args.skip_threshold:
            skipped_iqae = True
            q_idx = warmup["var_est_idx"]
            q_var = warmup["var_est_value"]
            q_est, q_ci = None, None
            q_queries = 0
            q_steps = 0
            print(f"Bracket width {warmup['bracket_width']} <= skip threshold "
                  f"{args.skip_threshold}: using MC result, IQAE skipped")
        elif warmup is not None:
            bracket = (warmup["lo"], warmup["hi"])
            print(f"Running quantum VaR (IQAE + bisection) within bracket "
                  f"[{bracket[0]}, {bracket[1]}] ...")
            q_idx, q_var, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk(
                grid_points, probs, ALPHA,
                epsilon=args.epsilon,
                alpha_fail=args.alpha_fail,
                bracket=bracket,
            )
        else:
            print(f"Running quantum VaR (IQAE + full bisection over {2**NUM_QUBITS} bins) ...")
            q_idx, q_var, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk(
                grid_points, probs, ALPHA,
                epsilon=args.epsilon,
                alpha_fail=args.alpha_fail,
                bracket=None,
            )

        # --- Classical MC VaR (from raw loss samples) ---
        mc_var_value = mc["var_mc"]
        n_mc_total = mc["n_samples"]
        sorted_losses = np.sort(mc["loss_samples"])
        k_lo = max(0, int(scipy.stats.binom.ppf(0.005, n_mc_total, ALPHA)))
        k_hi = min(n_mc_total - 1, int(scipy.stats.binom.ppf(0.995, n_mc_total, ALPHA)))
        mc_ci = (float(sorted_losses[k_lo]), float(sorted_losses[k_hi]))

        # --- Results ---
        print()
        print("=" * 70)
        print("RESULTS COMPARISON")
        print("=" * 70)

        print()
        print(f"1) Classical Monte Carlo (raw {n_mc_total:,} loss samples)")
        print(f"   VaR at {ALPHA*100:.0f}%          : {mc_var_value:.6f}")
        print(f"   99% CI              : [{mc_ci[0]:.6f}, {mc_ci[1]:.6f}]")
        print(f"   Samples used        : {n_mc_total:,}")
        print(f"   Device              : {mc['device']}")

        print()
        print(f"2) Classical CDF on discretized PMF ({2**NUM_QUBITS} bins)")
        print(f"   VaR at {ALPHA*100:.0f}%          : {ref_var:.6f}")
        print(f"   Grid index          : {ref_idx}")
        print(f"   Discretization      : [{GRID_LO:.4f}, {GRID_HI:.4f}]")

        print()
        label = "warm-start bisection" if warmup is not None else "full bisection"
        print(f"3) Quantum IQAE + {label}")
        print(f"   VaR at {ALPHA*100:.0f}%          : {q_var:.6f}")
        print(f"   Grid index          : {q_idx}")
        if q_est is not None:
            print(f"   Tail P estimate     : {q_est:.6f}")
            print(f"   IQAE CI             : [{q_ci[0]:.6f}, {q_ci[1]:.6f}]")
        print(f"   Bisection steps     : {q_steps}")
        print(f"   Oracle queries      : {q_queries}")

        print()
        print("Deltas:")
        print(f"   MC vs Discretized CDF    : {abs(mc_var_value - ref_var):.6f}")
        print(f"   MC vs Quantum            : {abs(mc_var_value - q_var):.6f}")
        print(f"   Discretized vs Quantum   : {abs(ref_var - q_var):.6f}")

        baseline_steps = NUM_QUBITS - 2
        saved = max(0, baseline_steps - q_steps)
        print()
        print("-" * 70)
        print("COST BREAKDOWN")
        print("-" * 70)
        print(f"  Factor model dimension    : d={model.d}, M={model.M}")
        print(f"  MC samples (GPU/CPU)      : {n_mc_total:,}  ({mc['device']})")
        print(f"  MC cost                   : O(N*M*d) = O({n_mc_total}*{model.M}*{model.d})"
              f" = ~{n_mc_total * model.M * model.d:.2e} FLOPs")
        if warmup is not None:
            print(f"  Warm-start bracket        : [{warmup['lo']}, {warmup['hi']}]  "
                  f"(width={warmup['bracket_width']})")
        else:
            print(f"  Warm-start bracket        : disabled")
        print(f"  Bisection steps B         : {q_steps}  "
              f"(vs ~{baseline_steps} without warm start, saved ~{saved})")
        print(f"  Oracle queries (quantum)  : {q_queries}")
        print(f"  Classical MC convergence  : O(1/sqrt(N)) = O({1/np.sqrt(n_mc_total):.6f})")
        print(f"  IQAE convergence          : O(1/eps)     eps={args.epsilon}")
        print("=" * 70)

    # --- Optional epsilon sweep ---
    if args.sweep:
        print()
        print("Running IQAE epsilon sweep ...")
        epsilons = [0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]
        iqae_epsilon_sweep(epsilons, alpha_fail=args.alpha_fail)
