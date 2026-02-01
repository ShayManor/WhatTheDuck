#!/usr/bin/env python3
"""
sanity_qae_qasm_test.py

Qualitative + quantitative sanity test for qae_sweep_qasm.py:
  1) Verify the QASM state-prep matches the saved npz probabilities.
  2) Verify the comparator wiring by comparing "exact objective probability" vs expected CDF.
  3) Run IterativeAmplitudeEstimation (IAE) with "normal" hyperparameters and compare to exact.
  4) Run the VaR bisection using IAE and compare idx/var to the reference in the npz.

Key point: This script uses the *correct* EstimationProblem wiring:
    EstimationProblem(state_preparation=A, objective_qubits=[obj])
and does NOT build a custom GroverOperator.

Usage examples:
  python sanity_qae_qasm_test.py --module qae_sweep_qasm.py --indir build_qasm --dist beta --alpha 0.05 --method statevector --shots 2000
  python sanity_qae_qasm_test.py --module qae_sweep_qasm.py --indir build_qasm --dist beta --alpha 0.05 --method automatic  --shots 2000
"""

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sweepmod", str(path.resolve()))
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def fmt(x, w=10, p=6):
    return f"{x:{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, default="qae_sweep_qasm.py")
    ap.add_argument("--indir", type=str, default="build_qasm")
    ap.add_argument("--dist", type=str, default="beta")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--method", type=str, default="statevector",
                    help="Aer Sampler backend method: statevector / automatic / density_matrix / matrix_product_state ...")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=395)

    # "normal" AE hyperparameters (reasonable defaults for this toy problem)
    ap.add_argument("--epsilon", type=float, default=0.02, help="IAE epsilon_target (absolute)")
    ap.add_argument("--alpha_fail", type=float, default=0.05, help="IAE alpha (failure prob => 95% CI when 0.05)")
    ap.add_argument("--max_steps", type=int, default=32, help="Max bisection steps (>=7 is enough for 2^7 grid)")
    ap.add_argument("--prob_tol", type=float, default=None,
                    help="Optional tolerance for bisection comparisons. If not provided, computed from the distribution.")
    args = ap.parse_args()

    m = load_module(Path(args.module))
    indir = Path(args.indir)

    npz = indir / f"{args.dist}.data.npz"
    qasm = indir / f"{args.dist}.stateprep.qasm"
    if not npz.exists() or not qasm.exists():
        raise SystemExit(f"Missing artifacts: {npz} or {qasm}")

    blob = np.load(npz, allow_pickle=False)
    probs_npz = blob["probs"].astype(float)
    grid = blob["grid"].astype(float)
    n = len(probs_npz)
    num_qubits = int(round(math.log2(n)))
    if 2 ** num_qubits != n:
        raise SystemExit(f"npz probs length {n} is not a power of two")

    ref_idx = int(blob["ref_idx"])
    ref_var = float(blob["ref_var"])

    # ---- Load stateprep circuit from the module
    stateprep = m._load_stateprep_qasm(qasm)

    # ---- 1) Verify QASM distribution matches NPZ (via Statevector)
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(stateprep)
    probs_qasm = sv.probabilities()

    l1 = float(np.sum(np.abs(probs_qasm - probs_npz)))
    print("\n[1] State-prep validation")
    print(f"    dist={args.dist}  n={n}  num_qubits={num_qubits}")
    print(f"    L1(probs_qasm - probs_npz) = {l1:.3e}")
    print(f"    argmax(qasm)={int(np.argmax(probs_qasm))}  p={float(np.max(probs_qasm)):.6f}")
    print(f"    argmax(npz )={int(np.argmax(probs_npz))}  p={float(np.max(probs_npz)):.6f}")

    # ---- Helpers: expected CDF and exact objective probability
    cdf = np.cumsum(probs_npz)

    def expected_cdf(t: int) -> float:
        # P(x < t) with x being the integer index (matches your comparator geq=False wiring)
        t = int(t)
        t = max(0, min(t, n))
        return float(np.sum(probs_npz[:t]))

    def exact_p1(t: int) -> float:
        A, obj = m._build_threshold_stateprep(
            stateprep_asset_only=stateprep,
            num_asset_qubits=num_qubits,
            threshold_index=int(t),
        )
        svA = Statevector.from_instruction(A)
        return float(svA.probabilities([obj])[1])

    # ---- 2) Verify comparator wiring (exact == expected)
    print("\n[2] Comparator wiring check (exact statevector vs expected CDF)")
    test_thresholds = sorted(set([0, 1, 2, 7, 8, 9, 11, 15, 31, 32, 63, 64, 96, 120, ref_idx, ref_idx + 1]))
    test_thresholds = [t for t in test_thresholds if 0 <= t <= n]
    print("    t    expected_cdf      exact_p1        abs_err")
    for t in test_thresholds:
        expv = expected_cdf(t)
        ex = exact_p1(t)
        print(f"  {t:3d}  {fmt(expv)}  {fmt(ex)}  {fmt(abs(ex-expv), w=10, p=6)}")

    # ---- 3) IAE test (correct EstimationProblem wiring)
    print("\n[3] IAE accuracy check (vs exact)")
    from qiskit_aer.primitives import Sampler
    backend_options = {"method": args.method, "seed_simulator": int(args.seed)}
    run_options = {"shots": int(args.shots)}
    sampler = Sampler(backend_options=backend_options, run_options=run_options)

    try:
        from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
    except Exception:
        from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem  # type: ignore

    def iae_p1(t: int):
        A, obj = m._build_threshold_stateprep(
            stateprep_asset_only=stateprep,
            num_asset_qubits=num_qubits,
            threshold_index=int(t),
        )
        problem = EstimationProblem(
            state_preparation=A,
            objective_qubits=[obj],
        )
        iae = IterativeAmplitudeEstimation(
            epsilon_target=float(args.epsilon),
            alpha=float(args.alpha_fail),
            sampler=sampler,
        )
        res = iae.estimate(problem)
        p_hat = float(getattr(res, "estimation", np.nan))
        ci = getattr(res, "confidence_interval", (np.nan, np.nan))
        cost = getattr(res, "num_oracle_queries", getattr(res, "num_queries", 0))
        powers = getattr(res, "powers", None)
        return p_hat, (float(ci[0]), float(ci[1])), int(cost), powers

    print("    t    exact_p1       iae_p_hat      abs_err      CI_low       CI_high      cost   powers")
    for t in [15, 31, 63]:
        ex = exact_p1(t)
        p_hat, (lo, hi), cost, powers = iae_p1(t)
        print(f"  {t:3d}  {fmt(ex)}  {fmt(p_hat)}  {fmt(abs(p_hat-ex), w=10, p=6)}  {fmt(lo)}  {fmt(hi)}  {cost:5d}  {powers}")

    # ---- 4) VaR bisection test (IAE-driven, should recover ref_idx/ref_var)
    alpha_target = float(args.alpha)  # IMPORTANT: do NOT scale by 0.01

    if args.prob_tol is None:
        # Reasonable default: half the smallest nonzero CDF step, but at least 0.005
        diffs = np.diff(cdf)
        diffs = diffs[diffs > 0]
        min_gap = float(np.min(diffs)) if len(diffs) else 0.0
        prob_tol = max(0.5 * min_gap, 0.005)
    else:
        prob_tol = float(args.prob_tol)

    print("\n[4] VaR bisection using IAE estimates")
    print(f"    alpha_target={alpha_target:.6f}  epsilon={args.epsilon:.4f}  alpha_fail={args.alpha_fail:.4f}  shots={args.shots}  prob_tol={prob_tol:.6f}")
    print(f"    reference: ref_idx={ref_idx}  ref_var={ref_var:.6f}")

    lo, hi = 0, n - 1
    total_cost = 0
    for step in range(int(args.max_steps)):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        p_hat, (ci_lo, ci_hi), cost, _ = iae_p1(mid)
        total_cost += cost

        # Decision logic (same style as your solve_var_bisect_quantum)
        if alpha_target > ci_hi + prob_tol:
            lo = mid + 1
            decision = "lo=mid+1 (alpha > CI_high)"
        elif alpha_target < ci_lo - prob_tol:
            hi = mid
            decision = "hi=mid   (alpha < CI_low)"
        else:
            if p_hat < alpha_target:
                lo = mid + 1
                decision = "lo=mid+1 (p_hat < alpha)"
            else:
                hi = mid
                decision = "hi=mid   (p_hat >= alpha)"

        print(f"    step={step:2d} mid={mid:3d}  p_hat={p_hat:.6f}  CI=[{ci_lo:.6f},{ci_hi:.6f}]  -> {decision}")

    idx_hat = lo
    var_hat = float(grid[idx_hat])
    rel_err = abs(var_hat - ref_var) / (abs(ref_var) + 1e-12)
    print("\n    RESULT")
    print(f"    idx_hat={idx_hat}  var_hat={var_hat:.6f}")
    print(f"    idx_err={idx_hat - ref_idx:+d}  rel_err={rel_err:.6f}  total_oracle_queries~{total_cost}")

    # Also show what the *true* CDF is at idx_hat and neighbors (discrete effects)
    def cdf_at(i: int) -> float:
        i = max(0, min(i, n))
        return float(np.sum(probs_npz[:i]))

    print("\n    Discrete CDF around idx_hat:")
    for i in [idx_hat - 2, idx_hat - 1, idx_hat, idx_hat + 1]:
        if 0 <= i <= n:
            print(f"      i={i:3d}  CDF(i)={cdf_at(i):.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
