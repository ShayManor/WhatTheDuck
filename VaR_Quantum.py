"""
Quantum Value at Risk (VaR) via Iterative Quantum Amplitude Estimation (IQAE)

Gaussian asset return distribution (toy model) over a fixed time horizon.

The quantum circuit estimates VaR by:
  1. State preparation  -- encode discretised Gaussian P&L PMF as qubit amplitudes
                           so |psi> = sum_i sqrt(p_i) |i>
  2. Threshold oracle   -- flip an indicator qubit when asset_index < threshold,
                           marking tail-loss events
  3. IQAE               -- estimate P(indicator=1) = CDF(threshold) with O(1/eps)
                           query complexity (quadratic speedup over classical MC)
  4. Bisection search   -- binary search over threshold indices until
                           CDF(threshold) ~ alpha, yielding the VaR grid value

Optional hybrid pipeline:
  0. MC warm start      -- GPU/QMC Monte Carlo to get a tight bracket [lo, hi]
                           around VaR index, reducing bisection steps B.
                           Total quantum cost ~ B x C(eps) x S(n).
"""

import argparse
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from classiq import *
from classiq.applications.iqae.iqae import IQAE

# ---------------------------------------------------------------------------
# Distribution & VaR parameters
# ---------------------------------------------------------------------------
NUM_QUBITS = 7          # 2^7 = 128 grid points
MU = 0.15               # mean daily P&L return
SIGMA = 0.20            # std-dev of daily P&L return
STUDENT_T_DF = 4        # degrees of freedom for heavy tails
SKEW_NC = -1.0          # non-centrality for skew (negative -> heavier left tail)
ALPHA = 0.05            # confidence level: P(P&L < VaR) = alpha
TOLERANCE = ALPHA / 10  # bisection convergence tolerance

# IQAE tuning
IQAE_EPSILON = 0.05     # target estimation precision
IQAE_ALPHA_FAIL = 0.01  # failure probability for confidence interval

# QAE (QPE-based) tuning
QAE_M = 4               # phase register size (precision ~ 1/2^m)
QAE_SHOTS = 1000        # shots per QAE run

# Grid spans +/- 3 sigma around the mean
GRID_LO = MU - 3 * SIGMA
GRID_HI = MU + 3 * SIGMA

#### Build student t distribution ####
# heavier tails - thats what the reality looks like
import numpy as np
import scipy.stats





def build_tail_focused_pmf_from_dist(
    dist,
    num_qubits: int,
    tail_alpha: float = 0.01,      # VaR tail
    tail_mass: float = 0.30,       # allocate this probability mass region to "tail zoom"
    tail_bin_frac: float = 0.70,   # fraction of bins spent in [0, tail_mass]
    clip_mass: float = 1e-6,       # avoid infinite quantiles
    pmf_mode: str = "cdf_diff",    # "cdf_diff" (best) or "pdf_weight"
):
    """
    Build a non-uniform grid focused on the left tail of the distribution.

    - dist: a scipy.stats distribution object (e.g., scipy.stats.norm(loc=..., scale=...))
    - tail_mass: how much CDF mass near 0 you want to zoom into (e.g., 0.30 means up to 30th percentile)
    - tail_bin_frac: fraction of bins allocated to that tail region
    - pmf_mode:
        * "cdf_diff": probs[i] = F(x_{i+1}) - F(x_i)  (probability mass per bin interval)  [recommended]
        * "pdf_weight": probs[i] ~ f(x_i) * delta_x   (ok but less stable on nonuniform grids)
    Returns: (grid_points, probs) where len(probs)=2^num_qubits, sum(probs)=1
    """
    N = 2 ** num_qubits

    # Split bins: more bins for tail region, fewer for the rest
    N_tail = int(np.round(N * tail_bin_frac))
    N_tail = max(4, min(N - 4, N_tail))
    N_body = N - N_tail

    # Define quantile ranges in CDF space
    # Tail zoom: [clip_mass, tail_mass]
    # Body:      [tail_mass, 1-clip_mass]
    u_tail = np.linspace(clip_mass, tail_mass, N_tail, endpoint=False)
    u_body = np.linspace(tail_mass, 1.0 - clip_mass, N_body)

    u = np.concatenate([u_tail, u_body])
    u = np.clip(u, clip_mass, 1.0 - clip_mass)

    # Grid points at those quantiles
    x = dist.ppf(u)

    # Ensure monotonic and finite
    x = np.nan_to_num(x, neginf=dist.ppf(clip_mass), posinf=dist.ppf(1.0 - clip_mass))
    x = np.sort(x)

    # Build probabilities on this nonuniform grid
    if pmf_mode == "cdf_diff":
        # Define bin edges midway between points; use CDF differences for mass
        edges = np.empty(N + 1)
        edges[1:-1] = 0.5 * (x[1:] + x[:-1])
        # extend edges slightly beyond extremes using quantiles
        edges[0] = dist.ppf(clip_mass)
        edges[-1] = dist.ppf(1.0 - clip_mass)

        cdf_edges = dist.cdf(edges)
        probs = np.diff(cdf_edges)
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()
        return x, probs.tolist()

    elif pmf_mode == "pdf_weight":
        # approximate mass around each point with local delta_x
        dx = np.empty_like(x)
        dx[1:-1] = 0.5 * (x[2:] - x[:-2])
        dx[0] = x[1] - x[0]
        dx[-1] = x[-1] - x[-2]
        pdf = dist.pdf(x)
        probs = pdf * dx
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()
        return x, probs.tolist()

    else:
        raise ValueError("pmf_mode must be 'cdf_diff' or 'pdf_weight'")






# ---------------------------------------------------------------------------
# Build discretised Student-t distribution on a uniform grid
# ---------------------------------------------------------------------------
def build_uniform_pmf_from_dist(dist, num_qubits, lo, hi):
    """Return (grid, probs) on a uniform grid where probs is a normalized PMF list."""
    n = 2 ** num_qubits
    grid = np.linspace(lo, hi, n)
    pdf = dist.pdf(grid)
    probs = (pdf / pdf.sum()).tolist()
    return grid, probs

def plot_distribution(grid, pmf, dist, output_path="results/distribution.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = dist.pdf(grid)
    plt.figure(figsize=(8, 4))
    plt.plot(grid, pdf, label="PDF")
    plt.step(grid, pmf, where="mid", label="PMF (uniform grid)")
    plt.xlabel("P&L")
    plt.ylabel("density / mass")
    plt.title("Student-t (skewed) on uniform grid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved distribution plot to {output_path}")

dist = scipy.stats.nct(df=STUDENT_T_DF, nc=SKEW_NC, loc=MU, scale=SIGMA)

grid_points, probs = build_uniform_pmf_from_dist(
    dist=dist,
    num_qubits=NUM_QUBITS,
    lo=GRID_LO,
    hi=GRID_HI,
)

# Tail-focused discretization (disabled for uniform grid baseline):
# grid_points, probs = build_tail_focused_pmf_from_dist(
#     dist=dist,
#     num_qubits=NUM_QUBITS,
#     tail_alpha=ALPHA,
#     tail_mass=0.30,        # zoom into bottom 30% mass
#     tail_bin_frac=0.70,    # spend 70% of bins there
#     clip_mass=1e-6,
#     pmf_mode="cdf_diff",
# )


# Module-level so Classiq @qfunc / @qperm decorators can capture them
#grid_points, probs = build_gaussian_pmf(MU, SIGMA, NUM_QUBITS, GRID_LO, GRID_HI)

# Mutable threshold index consumed by the oracle; updated during bisection
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
# Monte Carlo warm start
# ---------------------------------------------------------------------------
def mc_warm_start(grid, pmf, alpha, n_samples=10000, method="qmc",
                  confidence=0.99, seed=42):
    """
    Sample from the discrete PMF and return a tight [lo, hi] bracket for
    the VaR index.  Reduces bisection steps from log2(N) to log2(hi-lo).

    Methods
    -------
    plain : standard MC via numpy.random.choice
    qmc   : Quasi-MC with Sobol sequences — O(1/N) convergence
    gpu   : GPU-accelerated via PyTorch torch.multinomial (CUDA if available)

    Returns
    -------
    dict with keys: lo, hi, var_est_idx, var_est_value, n_samples,
                    method, device, bracket_width
    """
    cdf = np.cumsum(pmf)

    if method == "qmc":
        from scipy.stats import qmc as sp_qmc
        sampler = sp_qmc.Sobol(d=1, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n_samples, 2))))
        u = sampler.random_base2(m).flatten()          # 2^m points in [0,1)
        indices = np.searchsorted(cdf, u, side="left")  # inverse CDF transform
        indices = np.clip(indices, 0, len(grid) - 1)
        device = "cpu"
        actual_n = len(u)

    elif method == "gpu":
        device = "cpu"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            p_t = torch.tensor(pmf, device=device, dtype=torch.float32)
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            indices = torch.multinomial(
                p_t, n_samples, replacement=True, generator=g
            ).cpu().numpy()
        except ImportError:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(grid), size=n_samples, p=pmf)
        actual_n = len(indices)

    else:  # "plain"
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(grid), size=n_samples, p=pmf)
        device = "cpu"
        actual_n = len(indices)

    # --- Point estimate: alpha-quantile of sampled indices ---
    sorted_idx = np.sort(indices)
    k_center = max(0, min(int(np.floor(alpha * actual_n)), actual_n - 1))
    var_est_idx = int(sorted_idx[k_center])

    # --- Bracket via binomial order-statistic CI ---
    # The alpha-quantile is the k-th order statistic where k ~ Binom(n, alpha).
    # A (confidence)-level CI for the quantile is [X_(j_lo), X_(j_hi)].
    beta = 1.0 - confidence
    j_lo = max(0, int(scipy.stats.binom.ppf(beta / 2, actual_n, alpha)))
    j_hi = min(actual_n - 1, int(scipy.stats.binom.ppf(1 - beta / 2, actual_n, alpha)))

    lo_idx = max(0, int(sorted_idx[j_lo]))
    hi_idx = min(len(grid) - 1, int(sorted_idx[j_hi]))

    return {
        "lo": lo_idx,
        "hi": hi_idx,
        "var_est_idx": var_est_idx,
        "var_est_value": float(grid[var_est_idx]),
        "n_samples": actual_n,
        "method": method,
        "device": device,
        "bracket_width": hi_idx - lo_idx,
    }


# ---------------------------------------------------------------------------
# Quantum circuit (Classiq)
# ---------------------------------------------------------------------------
@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    """Full A operator: load distribution then apply threshold oracle."""
    load_distribution(asset=asset)
    payoff(asset=asset, ind=ind)


@qfunc
def load_distribution(asset: QNum):
    """Encode the Gaussian PMF into qubit amplitudes via Classiq's state-prep."""
    inplace_prepare_state(probs, bound=0, target=asset)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    """Threshold oracle: flip |ind> when asset register < THRESHOLD_INDEX."""
    ind ^= asset < THRESHOLD_INDEX


# ---------------------------------------------------------------------------
# IQAE helpers
# ---------------------------------------------------------------------------
def run_iqae_detailed(iqae, epsilon, alpha_fail):
    """Run a single IQAE call and return estimation + resource metrics."""
    result = iqae.run(epsilon=epsilon, alpha=alpha_fail)

    estimation = float(result.estimation)
    ci_low = float(result.confidence_interval[0])
    ci_high = float(result.confidence_interval[1])

    # Extract per-iteration resource usage
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
# IQAE tail-probability estimation
# ---------------------------------------------------------------------------
def estimate_tail_probability(threshold_index, epsilon=IQAE_EPSILON,
                              alpha_fail=IQAE_ALPHA_FAIL):
    """
    Use IQAE to estimate P(asset < threshold_index).

    Returns
    -------
    estimation : float
        Point estimate of the tail probability.
    confidence_interval : tuple[float, float]
        (ci_low, ci_high) from IQAE.
    resources : dict
        shots_total, grover_calls, ks_used from the IQAE run.
    """
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
# Quantum VaR via bisection
# ---------------------------------------------------------------------------
def quantum_value_at_risk(grid, pmf, alpha, tolerance=TOLERANCE,
                          epsilon=IQAE_EPSILON, alpha_fail=IQAE_ALPHA_FAIL,
                          bracket=None):
    """
    Compute Value at Risk using IQAE + bisection search.

    When *bracket* is None, uses the original search pattern (start at n//4,
    halve step each round — backward compatible).

    When *bracket* = (lo, hi) is provided (e.g. from mc_warm_start), runs a
    proper lo/hi bisection within the bracket.  This needs only log2(hi-lo)
    IQAE calls instead of log2(n).

    Returns
    -------
    var_index : int
    var_value : float
    final_estimate : float | None
    confidence_interval : tuple[float, float] | None
    total_oracle_queries : int
    bisection_steps : int
    """
    n = len(grid)
    total_oracle_queries = 0
    bisection_steps = 0
    est, ci = None, None

    if bracket is not None:
        # ------ Bracketed bisection (warm-started) ------
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

        # If bracket was already <= 1 wide, run one IQAE to confirm
        if est is None:
            est, ci, res = estimate_tail_probability(hi, epsilon, alpha_fail)
            total_oracle_queries += res["grover_calls"]
            bisection_steps += 1
            print(f"  confirm idx={hi}  P_hat={est:.6f}"
                  f"  ci=[{ci[0]:.6f}, {ci[1]:.6f}]"
                  f"  shots={res['shots_total']}  grover={res['grover_calls']}")

        return hi, float(grid[hi]), est, ci, total_oracle_queries, bisection_steps

    else:
        # ------ Original search pattern (no warm start) ------
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


############## QAE implementation (QPE-based)

@qfunc
def qae_space_transform(packed_vars: QArray[QBit]):
    """State preparation A, operating on packed_vars = [asset | indicator]."""
    asset = QNum(size=NUM_QUBITS)
    ind = QBit()
    bind(packed_vars, [asset, ind])
    inplace_prepare_state(probs, bound=0, target=asset)
    bind([asset, ind], packed_vars)


@qfunc
def qae_oracle(packed_vars: QArray[QBit]):
    """Phase oracle that flips the phase of "good" states."""
    asset = QNum(size=NUM_QUBITS)
    ind = QBit()
    bind(packed_vars, [asset, ind])
    payoff(asset=asset, ind=ind)
    Z(ind)
    payoff(asset=asset, ind=ind)
    bind([asset, ind], packed_vars)


def build_qae_main(m: int):
    @qfunc
    def main(
        packed: Output[QArray[QBit, NUM_QUBITS + 1]],
        phase: Output[QNum[m]],
    ):
        allocate(packed)
        allocate(phase)
        amplitude_estimation(
            oracle=qae_oracle,
            space_transform=qae_space_transform,
            phase=phase,
            packed_vars=packed,
        )

    return main


def _phase_counts_to_amplitude(counts_by_phase, m):
    total = sum(counts_by_phase.values())
    if total == 0:
        return 0.0, 0.0
    est = 0.0
    var = 0.0
    for y, c in counts_by_phase.items():
        theta = y / (2**m)
        a = math.sin(math.pi * theta) ** 2
        est += a * c
    est /= total
    for y, c in counts_by_phase.items():
        theta = y / (2**m)
        a = math.sin(math.pi * theta) ** 2
        var += (a - est) ** 2 * c
    var /= max(1, total - 1)
    return est, var


def run_qae_detailed(m, shots, seed=42):
    qae_main = build_qae_main(m)
    qprog = synthesize(
        qae_main,
        constraints=Constraints(max_width=28),
        preferences=Preferences(machine_precision=NUM_QUBITS),
    )
    set_quantum_program_execution_preferences(
        qprog, ExecutionPreferences(num_shots=shots, random_seed=seed)
    )
    job = execute(qprog)
    details = job.get_sample_result()

    parsed = details.parsed_counts_of_outputs("phase")
    if not parsed:
        return {
            "estimation": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "shots_total": 0,
            "grover_calls": 0,
            "phase_counts": {},
        }

    phase_counts = {}
    for s in parsed:
        phase_val = int(s.state["phase"])
        phase_counts[phase_val] = phase_counts.get(phase_val, 0) + int(s.shots)

    shots_total = sum(phase_counts.values())
    est, var = _phase_counts_to_amplitude(phase_counts, m)
    se = math.sqrt(var / shots_total) if shots_total > 0 else 0.0
    ci_low = max(0.0, est - 1.96 * se)
    ci_high = min(1.0, est + 1.96 * se)

    return {
        "estimation": est,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "shots_total": shots_total,
        "grover_calls": (2**m - 1) * shots_total,
        "phase_counts": phase_counts,
    }


def estimate_tail_probability_qae(threshold_index, m=QAE_M, shots=QAE_SHOTS, seed=42):
    """Use QAE (QPE-based) to estimate P(asset < threshold_index)."""
    global THRESHOLD_INDEX
    THRESHOLD_INDEX = int(threshold_index)

    res = run_qae_detailed(m=m, shots=shots, seed=seed)
    ci = (res["ci_low"], res["ci_high"])
    return res["estimation"], ci, res


def quantum_value_at_risk_qae(
    grid,
    pmf,
    alpha,
    tolerance=TOLERANCE,
    m=QAE_M,
    shots=QAE_SHOTS,
    seed=42,
    bracket=None,
):
    """
    Compute VaR using QAE (QPE-based) + bisection search.
    """
    n = len(grid)
    total_oracle_queries = 0
    bisection_steps = 0
    est, ci = None, None

    if bracket is not None:
        lo, hi = bracket
        while hi - lo > 1:
            mid = (lo + hi) // 2
            est, ci, res = estimate_tail_probability_qae(
                mid, m=m, shots=shots, seed=seed
            )
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
            est, ci, res = estimate_tail_probability_qae(
                hi, m=m, shots=shots, seed=seed
            )
            total_oracle_queries += res["grover_calls"]
            bisection_steps += 1
            print(f"  confirm idx={hi}  P_hat={est:.6f}"
                  f"  ci=[{ci[0]:.6f}, {ci[1]:.6f}]"
                  f"  shots={res['shots_total']}  grover={res['grover_calls']}")

        return hi, float(grid[hi]), est, ci, total_oracle_queries, bisection_steps

    index = n // 4
    search_size = index // 2

    est, ci, res = estimate_tail_probability_qae(
        index, m=m, shots=shots, seed=seed
    )
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
        est, ci, res = estimate_tail_probability_qae(
            index, m=m, shots=shots, seed=seed
        )
        total_oracle_queries += res["grover_calls"]
        bisection_steps += 1
        print(f"  idx={index:<4d}  P_hat={est:.6f}  ci=[{ci[0]:.6f}, {ci[1]:.6f}]"
              f"  shots={res['shots_total']}  grover={res['grover_calls']}")

    return index, float(grid[index]), est, ci, total_oracle_queries, bisection_steps









# ---------------------------------------------------------------------------
# IQAE epsilon sweep — measure scaling of queries vs precision
# ---------------------------------------------------------------------------
def iqae_epsilon_sweep(epsilons, alpha_fail=IQAE_ALPHA_FAIL,
                       output_path="results/iqae_epsilon_sweep.csv"):
    """
    Sweep IQAE precision (epsilon) and record estimation error vs oracle queries.

    Fixes the threshold at the classical VaR index so every run estimates the
    same true probability.  As epsilon shrinks the estimation gets tighter but
    grover_calls (oracle queries) grow — demonstrating O(1/eps) scaling.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Fix threshold at the classical VaR index
    ref_idx, _ = classical_var(grid_points, probs, ALPHA)
    global THRESHOLD_INDEX
    THRESHOLD_INDEX = int(ref_idx)
    a_true = sum(probs[:THRESHOLD_INDEX])

    # Build IQAE instance once (synthesises the circuit)
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

    rows = []
    for eps in epsilons:
        res = run_iqae_detailed(iqae, epsilon=eps, alpha_fail=alpha_fail)
        abs_error = abs(res["estimation"] - a_true)

        row = {
            "epsilon": eps,
            "alpha_fail": alpha_fail,
            "a_true": a_true,
            "a_hat": res["estimation"],
            "abs_error": abs_error,
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
            "shots_total": res["shots_total"],
            "grover_calls": res["grover_calls"],
            "ks_used": ";".join(str(k) for k in res["ks_used"] if k is not None),
        }
        rows.append(row)
        print(
            f"  eps={eps:>5.3f}  a_hat={res['estimation']:.6f}  "
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
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Quantum VaR with optional MC/QMC warm start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python VaR_Quantum.py                           # no warm start (original)
  python VaR_Quantum.py --mc-method qmc           # Sobol QMC warm start
  python VaR_Quantum.py --mc-method gpu --mc-samples 50000
  python VaR_Quantum.py --mc-method qmc --skip-threshold 2
  python VaR_Quantum.py --sweep                   # run epsilon sweep
""",
    )
    # MC warm start
    p.add_argument("--mc-method",
                   choices=["none", "plain", "qmc", "gpu"], default="none",
                   help="MC warm-start method (default: none)")
    p.add_argument("--mc-samples", type=int, default=10000,
                   help="number of MC samples for warm start (default: 10000)")
    p.add_argument("--mc-confidence", type=float, default=0.99,
                   help="CI confidence for bracket (default: 0.99)")
    p.add_argument("--skip-threshold", type=int, default=1,
                   help="skip IQAE if bracket width <= this (default: 1)")
    # IQAE
    p.add_argument("--epsilon", type=float, default=IQAE_EPSILON,
                   help=f"IQAE precision target (default: {IQAE_EPSILON})")
    p.add_argument("--alpha-fail", type=float, default=IQAE_ALPHA_FAIL,
                   help=f"IQAE failure probability (default: {IQAE_ALPHA_FAIL})")
    # QAE (QPE-based)
    p.add_argument("--use-qae", action="store_true",
                   help="use QAE (QPE-based) instead of IQAE")
    p.add_argument("--qae-m", type=int, default=QAE_M,
                   help=f"QAE phase register size m (default: {QAE_M})")
    p.add_argument("--qae-shots", type=int, default=QAE_SHOTS,
                   help=f"QAE shots per run (default: {QAE_SHOTS})")
    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-dist", action="store_true",
                   help="save distribution plot to results/distribution.png")
    p.add_argument("--sweep", action="store_true",
                   help="run IQAE epsilon sweep after VaR computation")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.plot_dist:
        plot_distribution(grid_points, probs, dist)

    # --- Classical reference ---
    ref_idx, ref_var = classical_var(grid_points, probs, ALPHA)
    print(f"Classical VaR at {ALPHA*100:.0f}%: {ref_var:.6f}  (index {ref_idx})")
    print()

    # --- Optional MC warm start ---
    warmup = None
    bracket = None
    skipped_iqae = False

    if args.mc_method != "none":
        print(f"MC warm start ({args.mc_method}, n={args.mc_samples}) ...")
        warmup = mc_warm_start(
            grid_points, probs, ALPHA,
            n_samples=args.mc_samples,
            method=args.mc_method,
            confidence=args.mc_confidence,
            seed=args.seed,
        )
        print(f"  bracket     : [{warmup['lo']}, {warmup['hi']}]  "
              f"(width={warmup['bracket_width']})")
        print(f"  MC VaR est  : {warmup['var_est_value']:.6f}  "
              f"(index {warmup['var_est_idx']})")
        print(f"  device      : {warmup['device']}")
        print(f"  samples used: {warmup['n_samples']}")
        print()

        if warmup["bracket_width"] <= args.skip_threshold:
            skipped_iqae = True
            q_idx = warmup["var_est_idx"]
            q_var = warmup["var_est_value"]
            q_est, q_ci = None, None
            q_queries = 0
            q_steps = 0
            print(f"Bracket width {warmup['bracket_width']} <= skip threshold "
                  f"{args.skip_threshold}: using MC result, IQAE skipped")
        else:
            bracket = (warmup["lo"], warmup["hi"])

    # --- Quantum VaR (IQAE or QAE + bisection) ---
    if not skipped_iqae:
        if args.use_qae:
            print("Running quantum VaR (QAE + bisection)"
                  + (f" within bracket [{bracket[0]}, {bracket[1]}]" if bracket else "")
                  + " ...")
            q_idx, q_var, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk_qae(
                grid_points, probs, ALPHA,
                m=args.qae_m,
                shots=args.qae_shots,
                seed=args.seed,
                bracket=bracket,
            )
        else:
            print("Running quantum VaR (IQAE + bisection)"
                  + (f" within bracket [{bracket[0]}, {bracket[1]}]" if bracket else "")
                  + " ...")
            q_idx, q_var, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk(
                grid_points, probs, ALPHA,
                epsilon=args.epsilon,
                alpha_fail=args.alpha_fail,
                bracket=bracket,
            )

    # --- Results ---
    print()
    print(f"Quantum  VaR at {ALPHA*100:.0f}%: {q_var:.6f}  (index {q_idx})")
    if q_est is not None:
        print(f"  Tail probability estimate : {q_est:.6f}")
        print(f"  Confidence interval       : [{q_ci[0]:.6f}, {q_ci[1]:.6f}]")
    print(f"  Abs error vs classical    : {abs(q_var - ref_var):.6f}")
    print(f"  Total oracle queries      : {q_queries}")
    print(f"  Bisection steps           : {q_steps}")
    if args.use_qae:
        print(f"  QAE phase register (m)    : {args.qae_m}")
        print(f"  QAE shots per run         : {args.qae_shots}")

    # --- Cost breakdown ---
    if warmup is not None:
        baseline_steps = 5  # empirical: log2(128/4) rounds without warm start
        saved = max(0, baseline_steps - q_steps)
        print()
        print("Cost breakdown (Total ~ B x C(eps) x S(n)):")
        print(f"  MC samples (cheap)        : {warmup['n_samples']}")
        print(f"  Bisection steps B         : {q_steps}  "
              f"(vs ~{baseline_steps} without warm start, saved ~{saved})")
        print(f"  Oracle queries            : {q_queries}")

    # --- Optional epsilon sweep ---
    if args.sweep:
        print()
        print("Running IQAE epsilon sweep ...")
        epsilons = [0.20, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
        iqae_epsilon_sweep(epsilons, alpha_fail=args.alpha_fail)
