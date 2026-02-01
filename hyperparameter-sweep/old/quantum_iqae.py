#!/usr/bin/env python3
"""
qae_sweep.py

Two modes:
  1) compile: Classiq -> OpenQASM state-prep only
  2) run:     Load QASM on GPU node, add threshold oracle, run IQAE+bisection, sweep algorithmic params

Key design:
- QASM is static; thresholds change during bisection.
- So we export ONLY state prep to QASM, and build the threshold oracle in Qiskit at runtime.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# ----------------------------
# Simple toy distributions
# ----------------------------
def make_grid(num_qubits: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    n = 2**num_qubits
    return np.linspace(lo, hi, n, endpoint=False)


def pmf_gaussian(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    # Discretize continuous PDF on grid by evaluating PDF then normalizing.
    x = grid
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    p = pdf / pdf.sum()
    return p.astype(float)


def pmf_lognormal(grid: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    # Map grid to positive support via shift; crude toy distribution.
    x = np.clip(grid - grid.min() + 1e-6, 1e-6, None)
    pdf = (1 / (x * sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((np.log(x) - mean) / sigma) ** 2)
    p = pdf / pdf.sum()
    return p.astype(float)


def build_pmf(dist: str, grid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "gaussian":
        # Reasonable defaults; adjust as needed.
        return pmf_gaussian(grid, mu=0.15, sigma=0.20)
    if dist == "lognormal":
        return pmf_lognormal(grid, mean=0.0, sigma=0.5)
    if dist == "bimodal":
        p1 = pmf_gaussian(grid, mu=-0.1, sigma=0.10)
        p2 = pmf_gaussian(grid, mu=+0.2, sigma=0.12)
        w = 0.45 + 0.1 * (rng.random() - 0.5)
        p = w * p1 + (1 - w) * p2
        return (p / p.sum()).astype(float)
    raise ValueError(f"Unknown --dist {dist}")


def discrete_var(grid: np.ndarray, probs: np.ndarray, alpha: float) -> Tuple[int, float]:
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, alpha, side="left"))
    idx = max(0, min(idx, len(grid) - 1))
    return idx, float(grid[idx])


# ----------------------------
# Compile: Classiq -> QASM
# ----------------------------
def cmd_compile(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(args.num_qubits, lo=args.grid_lo, hi=args.grid_hi)
    probs = build_pmf(args.dist, grid, args.seed)

    # Optional: compute reference VaR
    var_idx, var_val = discrete_var(grid, probs, args.alpha)

    # IMPORTANT: Do NOT use `from __future__ import annotations` in this file.
    # Classiq needs real typing objects for Output[QNum] etc (not strings).
    import classiq
    from classiq import Output, QNum, create_model, qfunc, synthesize, Preferences, QuantumFormat, prepare_state

    if args.login:
        # Use your existing auth behavior. The warning you saw is non-fatal.
        classiq.authenticate()

    probs_list = probs.tolist()

    @qfunc
    def main(asset: Output[QNum]):  # <-- keep as real annotation, not a string
        # State prep only, no oracle here
        prepare_state(probabilities=probs_list, bound=args.bound, out=asset)

    model = create_model(main)
    # QASM output is provided on the synthesized program (`qprog.qasm`). :contentReference[oaicite:11]{index=11}
    qprog = synthesize(
        model,
        preferences=Preferences(
            output_format=[QuantumFormat.QASM],
            optimization_level=args.opt_level,
        ),
    )

    qasm_path = outdir / "stateprep.qasm"
    qasm_path.write_text(qprog.qasm, encoding="utf-8")

    npz_path = outdir / "dist.npz"
    np.savez_compressed(npz_path, grid=grid, probs=probs, cdf=np.cumsum(probs))

    manifest = {
        "num_qubits": args.num_qubits,
        "dist": args.dist,
        "alpha": args.alpha,
        "seed": args.seed,
        "grid_lo": args.grid_lo,
        "grid_hi": args.grid_hi,
        "bound": args.bound,
        "opt_level": args.opt_level,
        "files": {
            "qasm": str(qasm_path.name),
            "npz": str(npz_path.name),
        },
        "reference": {
            "var_index": var_idx,
            "var_value": var_val,
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote:\n  {qasm_path}\n  {npz_path}\n  {outdir/'manifest.json'}")


# ----------------------------
# Run: Qiskit + Aer GPU + IQAE + sweep
# ----------------------------
def _load_manifest(indir: Path) -> Dict:
    return json.loads((indir / "manifest.json").read_text(encoding="utf-8"))


def _make_aer_gpu_backend():
    # Aer GPU runs with device="GPU" in AerSimulator.
    from qiskit_aer import AerSimulator

    backend = AerSimulator(method="statevector", device="GPU")
    return backend


def _build_problem_from_stateprep_qasm(
    stateprep_qasm: str,
    num_qubits: int,
    threshold_index_exclusive: int,
    reverse_bits: bool,
):
    """
    Builds an EstimationProblem where objective qubit is 1 iff (asset_index < threshold_index_exclusive).

    We implement the threshold via IntegerComparator in Qiskit at runtime.
    The only thing imported from QASM is the asset state-preparation circuit.
    """
    from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
    from qiskit.circuit.library import IntegerComparator

    # Import QASM as a circuit on `num_qubits` qubits
    stateprep = QuantumCircuit.from_qasm_str(stateprep_qasm)
    if stateprep.num_qubits != num_qubits:
        raise ValueError(f"QASM has {stateprep.num_qubits} qubits but manifest expects {num_qubits}")

    qr = QuantumRegister(num_qubits, "asset")
    obj = QuantumRegister(1, "obj")

    # IntegerComparator compares a quantum integer to a classical constant and writes to result qubit.
    # We use (asset < threshold_index_exclusive).
    comp = IntegerComparator(num_state_qubits=num_qubits, value=threshold_index_exclusive, geq=False)

    # Comparator needs total qubits = num_qubits (state) + 1 (result) + ancillas
    anc_n = comp.num_qubits - (num_qubits + 1)
    anc = AncillaRegister(anc_n, "anc") if anc_n > 0 else None

    qc = QuantumCircuit(qr, obj, anc) if anc is not None else QuantumCircuit(qr, obj)

    # Apply imported stateprep onto qr
    qc.compose(stateprep, qubits=list(qr), inplace=True)

    asset_wires = list(qr)
    if reverse_bits:
        asset_wires = list(reversed(asset_wires))

    # Append comparator: (asset_wires, obj, anc)
    wires = asset_wires + [obj[0]] + (list(anc) if anc is not None else [])
    qc.append(comp, wires)

    # EstimationProblem default good-state condition is objective qubits all-ones. :contentReference[oaicite:13]{index=13}
    from qiskit_algorithms import EstimationProblem

    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=[num_qubits],  # obj is after asset, ancillas excluded
    )
    return problem


def _pick_bit_order_by_cdf_check(
    stateprep_qasm: str, num_qubits: int, cdf: np.ndarray, test_thr: int = 13
) -> bool:
    """
    Endianness / bit-order sanity check:
    - Build problem with reverse_bits False/True
    - Compute exact p(obj=1) via Statevector for a single threshold
    - Choose mapping that matches classical CDF best
    """
    from qiskit.quantum_info import Statevector

    test_thr = max(1, min(test_thr, 2**num_qubits - 1))
    p_true = float(cdf[test_thr - 1])  # P(asset < test_thr) == CDF(test_thr-1)

    def p_from_problem(reverse_bits: bool) -> float:
        problem = _build_problem_from_stateprep_qasm(stateprep_qasm, num_qubits, test_thr, reverse_bits)
        sv = Statevector.from_instruction(problem.state_preparation)
        # objective qubit index among state qubits is `num_qubits`
        probs = sv.probabilities()
        # Sum probabilities where objective bit is 1 (in computational basis ordering)
        p = 0.0
        n_total = problem.state_preparation.num_qubits
        obj_wire = num_qubits  # obj position in the circuit qubit list (asset then obj then anc)

        for i, pr in enumerate(probs):
            # bit test: qiskit uses little-endian indexing for basis label; use bit mask
            if (i >> obj_wire) & 1:
                p += float(pr)
        return p

    p_fwd = p_from_problem(False)
    p_rev = p_from_problem(True)

    err_fwd = abs(p_fwd - p_true)
    err_rev = abs(p_rev - p_true)
    return err_rev < err_fwd  # True => reverse_bits


def _run_iqae(
    problem,
    epsilon: float,
    alpha_fail: float,
    seed: int,
):
    """
    Runs IterativeAmplitudeEstimation on the given EstimationProblem.
    Returns (p_hat, p_ci_low, p_ci_high, num_oracle_queries).
    """
    # Qiskit algorithms API + Aer primitive/simulation stack differs across versions.
    # This path targets modern qiskit-algorithms with primitives v2.
    from qiskit_algorithms import IterativeAmplitudeEstimation

    # Use AerSamplerV2 if available; otherwise fall back to backend-based execution.
    sampler = None
    try:
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

        sampler = AerSamplerV2(default_shots=None, seed=seed)
        # Workaround for option plumbing: set options via underlying backend. :contentReference[oaicite:14]{index=14}
        sampler._backend = _make_aer_gpu_backend()
    except Exception:
        sampler = None

    # Build IAE with robust signature handling
    import inspect

    kwargs = {"alpha": alpha_fail}
    sig = inspect.signature(IterativeAmplitudeEstimation)
    if "epsilon_target" in sig.parameters:
        kwargs["epsilon_target"] = epsilon
    elif "epsilon" in sig.parameters:
        kwargs["epsilon"] = epsilon
    else:
        raise RuntimeError("IterativeAmplitudeEstimation signature missing epsilon parameter")

    if sampler is not None and "sampler" in sig.parameters:
        kwargs["sampler"] = sampler

    iae = IterativeAmplitudeEstimation(**kwargs)
    res = iae.estimate(problem)

    # Result fields: estimation and confidence interval; query counts available. :contentReference[oaicite:15]{index=15}
    p_hat = float(res.estimation)
    ci = res.confidence_interval
    p_lo = float(ci[0]) if ci is not None else float("nan")
    p_hi = float(ci[1]) if ci is not None else float("nan")
    q = int(getattr(res, "num_oracle_queries", -1))
    return p_hat, p_lo, p_hi, q


def cmd_run(args: argparse.Namespace) -> None:
    indir = Path(args.indir)
    mani = _load_manifest(indir)

    num_qubits = int(mani["num_qubits"])
    alpha = float(mani["alpha"])

    npz = np.load(indir / mani["files"]["npz"])
    grid = npz["grid"].astype(float)
    probs = npz["probs"].astype(float)
    cdf = npz["cdf"].astype(float)

    stateprep_qasm = (indir / mani["files"]["qasm"]).read_text(encoding="utf-8")

    reverse_bits = _pick_bit_order_by_cdf_check(stateprep_qasm, num_qubits, cdf)
    print(f"Comparator bit-order: reverse_bits={reverse_bits}")

    # Reference VaR
    ref_idx = int(mani["reference"]["var_index"])
    ref_val = float(mani["reference"]["var_value"])

    eps_list = [float(x) for x in args.epsilon_list]
    af_list = [float(x) for x in args.alpha_fail_list]

    results = []
    for eps in eps_list:
        for af in af_list:
            lo, hi = 0, len(grid) - 1
            total_queries = 0

            # Bisection to find smallest idx with CDF(idx) >= alpha
            for _ in range(args.max_steps):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2

                # We want P(asset <= mid) >= alpha.
                # Our comparator is (asset < thr_exclusive), so set thr_exclusive = mid+1.
                thr_excl = mid + 1

                problem = _build_problem_from_stateprep_qasm(
                    stateprep_qasm=stateprep_qasm,
                    num_qubits=num_qubits,
                    threshold_index_exclusive=thr_excl,
                    reverse_bits=reverse_bits,
                )

                p_hat, p_lo, p_hi, q = _run_iqae(problem, epsilon=eps, alpha_fail=af, seed=args.seed)
                if q >= 0:
                    total_queries += q

                # Decision using confidence interval:
                # - if upper < alpha => need larger threshold (move right)
                # - if lower > alpha => threshold too large (move left)
                # - else interval straddles alpha => shrink by moving hi to mid (conservative)
                if not (math.isfinite(p_lo) and math.isfinite(p_hi)):
                    # If CI not available, fall back to point estimate
                    if p_hat < alpha:
                        lo = mid
                    else:
                        hi = mid
                else:
                    if p_hi < alpha:
                        lo = mid
                    elif p_lo > alpha:
                        hi = mid
                    else:
                        hi = mid

            est_idx = hi
            est_val = float(grid[est_idx])
            err = abs(est_val - ref_val)

            results.append(
                {
                    "epsilon": eps,
                    "alpha_fail": af,
                    "est_index": est_idx,
                    "est_var": est_val,
                    "ref_index": ref_idx,
                    "ref_var": ref_val,
                    "abs_error": err,
                    "oracle_queries_total": total_queries,
                }
            )
            print(
                f"eps={eps:g} alpha_fail={af:g}  est_var={est_val:.6f}  ref_var={ref_val:.6f}  "
                f"abs_err={err:.6f}  queries={total_queries}"
            )

    outpath = Path(args.out_csv)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV (no pandas dependency)
    import csv

    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"Wrote {outpath}")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compile", help="Compile state-prep to OpenQASM via Classiq")
    pc.add_argument("--outdir", required=True)
    pc.add_argument("--num-qubits", type=int, required=True)
    pc.add_argument("--dist", choices=["gaussian", "lognormal", "bimodal"], default="gaussian")
    pc.add_argument("--alpha", type=float, default=0.05)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--grid-lo", type=float, default=-1.0)
    pc.add_argument("--grid-hi", type=float, default=1.0)
    pc.add_argument("--bound", type=float, default=0.0, help="Classiq state-prep bound (fixed; not swept)")
    pc.add_argument("--opt-level", type=int, default=0, help="Classiq optimization level (fixed; not swept)")
    pc.add_argument("--login", action="store_true")
    pc.set_defaults(func=cmd_compile)

    pr = sub.add_parser("run", help="Run IQAE+bisection on GPU node and sweep algorithmic params")
    pr.add_argument("--indir", required=True)
    pr.add_argument("--epsilon-list", nargs="+", required=True)
    pr.add_argument("--alpha-fail-list", nargs="+", required=True)
    pr.add_argument("--max-steps", type=int, default=64)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out-csv", default="results/sweep.csv")
    pr.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
