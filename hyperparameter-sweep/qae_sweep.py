#!/usr/bin/env python3
"""
qae_sweep_qasm.py

Two modes:
  1) compile: uses Classiq to synthesize *state preparation only* and export OpenQASM.
  2) run: loads QASM + saved distribution data and sweeps only algorithmic params
          using Qiskit iterative amplitude estimation on Aer (GPU-capable).

Artifacts per distribution:
  - <name>.stateprep.qasm
  - <name>.data.npz  (grid, probs, ref_var, ref_idx, alpha, num_qubits, dist_params)

Notes:
  - This deliberately ignores "structural" synthesis params in the sweep.
  - The threshold oracle is built in Qiskit via IntegerComparator at runtime.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as st


# -----------------------------
# Distributions (same semantics)
# -----------------------------

def get_log_normal_probabilities(mu: float, sigma: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.exp(mu + sigma**2 / 2)
    var = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    std = np.sqrt(var)
    low = max(0.0, mean - 3 * std)
    high = mean + 3 * std
    x = np.linspace(low, high, num_points)
    pdf = st.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    p = pdf / np.sum(pdf)
    return x, p


def get_normal_probabilities(mu: float, sigma: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    low = mu - 4 * sigma
    high = mu + 4 * sigma
    x = np.linspace(low, high, num_points)
    pdf = st.norm.pdf(x, loc=mu, scale=sigma)
    p = pdf / np.sum(pdf)
    return x, p


def get_exponential_probabilities(lam: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = 1.0 / lam
    high = mean + 5 * mean
    x = np.linspace(0.0, high, num_points)
    pdf = st.expon.pdf(x, scale=1.0 / lam)
    p = pdf / np.sum(pdf)
    return x, p


def get_pareto_probabilities(alpha_shape: float, x_m: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    high = st.pareto.ppf(0.99, b=alpha_shape, scale=x_m)
    x = np.linspace(x_m, high, num_points)
    pdf = st.pareto.pdf(x, b=alpha_shape, scale=x_m)
    p = pdf / np.sum(pdf)
    return x, p


def get_mixture_probabilities(num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1, 5, num_points)
    pdf = 0.8 * st.norm.pdf(x, loc=1, scale=0.2) + 0.2 * st.norm.pdf(x, loc=3, scale=0.5)
    p = pdf / np.sum(pdf)
    return x, p


def get_beta_probabilities(a: float, b: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(1e-6, 1 - 1e-6, num_points)
    pdf = st.beta.pdf(x, a, b)
    p = pdf / np.sum(pdf)
    return x, p


def get_student_t_probabilities(df: float, loc: float, scale: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    low = st.t.ppf(0.001, df, loc=loc, scale=scale)
    high = st.t.ppf(0.999, df, loc=loc, scale=scale)
    x = np.linspace(low, high, num_points)
    pdf = st.t.pdf(x, df, loc=loc, scale=scale)
    p = pdf / np.sum(pdf)
    return x, p


def get_skew_normal_probabilities(alpha_skew: float, loc: float, scale: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    low = st.skewnorm.ppf(0.001, alpha_skew, loc=loc, scale=scale)
    high = st.skewnorm.ppf(0.999, alpha_skew, loc=loc, scale=scale)
    x = np.linspace(low, high, num_points)
    pdf = st.skewnorm.pdf(x, alpha_skew, loc=loc, scale=scale)
    p = pdf / np.sum(pdf)
    return x, p


def get_distribution(dist_name: str, num_points: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    if dist_name == "lognormal":
        return get_log_normal_probabilities(kwargs.get("mu", 0.7), kwargs.get("sigma", 0.13), num_points)
    if dist_name == "normal":
        return get_normal_probabilities(kwargs.get("mu", 0.0), kwargs.get("sigma", 1.0), num_points)
    if dist_name == "exponential":
        return get_exponential_probabilities(kwargs.get("lam", 1.0), num_points)
    if dist_name == "pareto":
        return get_pareto_probabilities(kwargs.get("shape", 2.5), kwargs.get("scale", 1.0), num_points)
    if dist_name == "mixture":
        return get_mixture_probabilities(num_points)
    if dist_name == "beta":
        return get_beta_probabilities(kwargs.get("a", 2.0), kwargs.get("b", 5.0), num_points)
    if dist_name == "student_t":
        return get_student_t_probabilities(kwargs.get("df", 3.0), kwargs.get("loc", 0.0), kwargs.get("scale", 1.0), num_points)
    if dist_name == "skew_normal":
        return get_skew_normal_probabilities(kwargs.get("skew", 4.0), kwargs.get("loc", 0.0), kwargs.get("scale", 1.0), num_points)
    raise ValueError(f"Unknown distribution: {dist_name}")


def compute_true_var(grid_points: np.ndarray, probs: np.ndarray, alpha: float) -> Tuple[float, int]:
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, alpha, side="left"))
    return float(grid_points[idx]), idx


# -----------------------------
# Compile mode (Classiq -> QASM)
# -----------------------------

def compile_stateprep_to_qasm(
    *,
    probs: np.ndarray,
    num_qubits: int,
    out_qasm_path: Path,
    debug_mode: bool = False,
    qasm3: bool = False,
    bound: float = 0.0,
) -> None:
    """
    Synthesize only |0> -> sum_x sqrt(p[x])|x> using Classiq inplace_prepare_state,
    then export OpenQASM (2 by default, optionally OpenQASM 3).
    """
    import classiq
    from classiq import (
        Output,
        Preferences,
        QNum,
        allocate,
        create_model,
        inplace_prepare_state,
        qfunc,
        synthesize,
    )

    probs_arr = np.asarray(probs, dtype=float)
    probs_arr = probs_arr / np.sum(probs_arr)
    probs_list = probs_arr.tolist()

    @qfunc
    def main(asset: Output[QNum]):
        allocate(num_qubits, asset)
        inplace_prepare_state(probs_list, bound, asset)  # Fixed: added bound argument

    model = create_model(main)
    prefs = Preferences(
        output_format=["qasm"],
        debug_mode=bool(debug_mode),
        qasm3=True if qasm3 else None,
    )
    qprog = synthesize(model, preferences=prefs)
    qasm_str = qprog.qasm

    out_qasm_path.parent.mkdir(parents=True, exist_ok=True)
    out_qasm_path.write_text(qasm_str, encoding="utf-8")



def cmd_compile(args: argparse.Namespace) -> None:
    if args.login:
        import classiq
        classiq.authenticate()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    num_points = 2 ** int(args.num_qubits)

    test_distributions: List[Dict[str, Any]] = [
        {"dist": "lognormal", "mu": 0.7, "sigma": 0.13},
        {"dist": "normal", "mu": 0.0, "sigma": 1.0},
        {"dist": "exponential", "lam": 1.0},
        {"dist": "pareto", "shape": 2.5, "scale": 1.0},
        {"dist": "student_t", "df": 3.0, "loc": 0.0, "scale": 1.0},
        {"dist": "skew_normal", "skew": 4.0, "loc": 0.0, "scale": 1.0},
        {"dist": "beta", "a": 2.0, "b": 5.0},
        {"dist": "mixture"},
    ]

    for spec in test_distributions:
        name = spec["dist"]
        grid, probs = get_distribution(name, num_points, **spec)
        ref_var, ref_idx = compute_true_var(grid, probs, args.alpha)

        qasm_path = outdir / f"{name}.stateprep.qasm"
        data_path = outdir / f"{name}.data.npz"

        compile_stateprep_to_qasm(
            probs=probs,
            num_qubits=args.num_qubits,
            out_qasm_path=qasm_path,
            debug_mode=args.debug_mode,
            qasm3=args.qasm3,
        )

        np.savez_compressed(
            data_path,
            grid=grid.astype(np.float64),
            probs=probs.astype(np.float64),
            alpha=float(args.alpha),
            ref_var=float(ref_var),
            ref_idx=int(ref_idx),
            num_qubits=int(args.num_qubits),
            dist_name=str(name),
            dist_params=json.dumps(spec),
        )

        print(f"[compile] {name}: wrote {qasm_path.name} and {data_path.name}")


# -----------------------------
# Run mode (QASM -> Qiskit AE)
# -----------------------------

@dataclass
class EstResult:
    p_hat: float
    ci_low: float
    ci_high: float
    cost_oracle_queries: int


def _load_stateprep_qasm(qasm_path: Path):
    import qiskit.qasm2 as qasm2
    qc = qasm2.load(str(qasm_path))

    # Remove measurements if any
    if qc.num_clbits > 0:
        qc.remove_final_measurements(inplace=True)

    # CRITICAL: Decompose custom gates to basis gates
    # Repeat until no more custom gates remain
    prev_depth = 0
    while qc.depth() != prev_depth:
        prev_depth = qc.depth()
        qc = qc.decompose()

    print(f"    QASM loaded and decomposed: depth={qc.depth()}, gates={qc.size()}")
    return qc
    """
    Load QASM and decompose to basis gates that Aer understands.
    """
    # try:
    #     import qiskit.qasm2 as qasm2
    #     qc = qasm2.load(str(qasm_path))
    # except (ImportError, AttributeError):
    #     from qiskit import QuantumCircuit
    #     qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    # from qiskit.transpiler import PassManager
    # from qiskit.transpiler.passes import Decompose, UnrollCustomDefinitions, BasisTranslator
    # from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
    #
    # qc = qasm2.load(str(qasm_path))
    #
    # # Remove measurements if any
    # if qc.num_clbits > 0:
    #     qc.remove_final_measurements(inplace=True)
    #
    # # Decompose all custom gates to Aer-supported basis gates
    # from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    # from qiskit_aer import AerSimulator
    #
    # backend = AerSimulator()
    # pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    # qc_decomposed = pm.run(qc)
    #
    # return qc_decomposed
    import qiskit.qasm2 as qasm2
    qc = qasm2.load(str(qasm_path))

    # Remove measurements if any
    if qc.num_clbits > 0:
        qc.remove_final_measurements(inplace=True)

    return qc


def _make_sampler_v2(device: str, method: str, seed: int, default_shots: int):
    """
    Create an Aer Sampler configured for GPU when available.
    """
    from qiskit_aer.primitives import Sampler

    backend_options = {"method": method, "seed_simulator": int(seed)}
    if device.upper() == "GPU":
        backend_options["device"] = "GPU"

    run_options = {"shots": int(default_shots)}

    return Sampler(backend_options=backend_options, run_options=run_options)
def _build_threshold_stateprep(
    stateprep_asset_only,
    num_asset_qubits: int,
    threshold_index: int,
):
    """
    Build state-prep + threshold oracle circuit, transpiled for Aer.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import IntegerComparator
    from qiskit_aer import AerSimulator

    comp = IntegerComparator(num_state_qubits=num_asset_qubits, value=threshold_index, geq=False)
    ancillas = comp.num_qubits - (num_asset_qubits + 1)
    if ancillas < 0:
        ancillas = 0

    qc = QuantumCircuit(num_asset_qubits + ancillas + 1)

    # Compose state preparation on the first num_asset_qubits wires
    qc.compose(stateprep_asset_only, qubits=list(range(num_asset_qubits)), inplace=True)

    asset_wires = list(range(num_asset_qubits))
    anc_wires = list(range(num_asset_qubits, num_asset_qubits + ancillas))
    obj_wire = [num_asset_qubits + ancillas]

    qc.append(comp, asset_wires + anc_wires + obj_wire)

    # Transpile the full circuit to basis gates Aer understands
    # backend = AerSimulator()
    # qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
    #
    # objective_index = obj_wire[0]
    # return qc_transpiled, objective_index
    objective_index = obj_wire[0]
    return qc, objective_index



def _estimate_tail_prob_iae(
    *,
    sampler_v2,
    stateprep_asset_only,
    num_asset_qubits: int,
    threshold_index: int,
    epsilon: float,
    alpha_fail: float,
) -> EstResult:
    print(
        f"    STATEPREP: depth={stateprep_asset_only.depth()}, gates={stateprep_asset_only.size()}, qubits={stateprep_asset_only.num_qubits}")
    """
    Run iterative amplitude estimation for this threshold index.
    Returns p_hat, CI, and oracle-query cost (as reported by Qiskit result when available).
    """
    # Newer package
    try:
        from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
    except Exception:
        # Older layouts (best-effort compatibility)
        from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem  # type: ignore

    A, obj = _build_threshold_stateprep(
        stateprep_asset_only=stateprep_asset_only,
        num_asset_qubits=num_asset_qubits,
        threshold_index=threshold_index,
    )
    print(f"    Circuit depth={A.depth()}, gates={A.size()}, qubits={A.num_qubits}")  # DEBUG

    from qiskit.circuit.library import GroverOperator

    # Explicitly build Grover operator with decomposed circuit
    A_decomposed = A.decompose().decompose().decompose()
    grover_op = GroverOperator(
        oracle=A_decomposed,
        state_preparation=A_decomposed,
        reflection_qubits=list(range(A.num_qubits - 1)),  # All except objective
    )

    problem = EstimationProblem(
        state_preparation=A,
        grover_operator=grover_op,
        objective_qubits=[obj],
        is_good_state=lambda bitstr: bitstr == "1",
    )

    iae = IterativeAmplitudeEstimation(
        epsilon_target=float(epsilon),
        alpha=float(alpha_fail),
        sampler=sampler_v2,
    )

    # API can be estimate() or run() depending on version.
    # if hasattr(iae, "estimate"):
    res = iae.estimate(problem)
    p_hat_raw = float(getattr(res, "estimation", np.nan))
    ci = getattr(res, "confidence_interval", (np.nan, np.nan))
    print(f"    DEBUG: p_hat={p_hat_raw:.6f}, CI=[{ci[0]:.6f}, {ci[1]:.6f}], threshold_idx={threshold_index}")
    print(f"    DEBUG: powers={getattr(res, 'powers', None)}, shots={getattr(res, 'shots', None)}")

    # else:
    #     res = iae.run(problem)  # type: ignore
    # Extract estimation and CI
    p_hat = float(getattr(res, "estimation", getattr(res, "estimation_processed", np.nan)))
    ci = getattr(res, "confidence_interval", None)
    if ci is None:
        # Some versions store it under "confidence_interval_processed"
        ci = getattr(res, "confidence_interval_processed", (0.0, 1.0))
    ci_low, ci_high = float(ci[0]), float(ci[1])

    # Cost: prefer official attribute if present
    cost = getattr(res, "num_oracle_queries", None)
    if cost is None:
        cost = getattr(res, "num_queries", None)
    if cost is None:
        cost = 0

    return EstResult(p_hat=p_hat, ci_low=ci_low, ci_high=ci_high, cost_oracle_queries=int(cost))


def solve_var_bisect_quantum(
    *,
    sampler_v2,
    stateprep_asset_only,
    grid_points: np.ndarray,
    probs: np.ndarray,
    alpha_target: float,
    epsilon: float,
    alpha_fail: float,
    prob_tol: float,
    max_steps: int,
) -> Tuple[float, int, int]:
    """
    Bisection on threshold index using IAE-estimated tail probability.

    Tail definition matches your existing code: p = P(asset_index < threshold_index).
    """
    n = len(grid_points)
    num_asset_qubits = int(round(math.log2(n)))
    if 2 ** num_asset_qubits != n:
        raise ValueError("grid_points length must be power of 2")

    lo, hi = 0, n - 1
    total_cost = 0

    # Cache A(threshold) circuits would require caching inside _estimate_tail_prob_iae;
    # keeping it simple here; main speedup is eliminating Classiq synthesis per call.
    for _ in range(int(max_steps)):
        if lo >= hi:
            break
        mid = (lo + hi) // 2

        est = _estimate_tail_prob_iae(
            sampler_v2=sampler_v2,
            stateprep_asset_only=stateprep_asset_only,
            num_asset_qubits=num_asset_qubits,
            threshold_index=mid,
            epsilon=epsilon,
            alpha_fail=alpha_fail,
        )
        total_cost += est.cost_oracle_queries

        if alpha_target > est.ci_high + prob_tol:
            lo = mid + 1
        elif alpha_target < est.ci_low - prob_tol:
            hi = mid
        else:
            if est.p_hat < alpha_target:
                lo = mid + 1
            else:
                hi = mid

    return float(grid_points[lo]), int(lo), int(total_cost)


def cmd_run(args: argparse.Namespace) -> None:
    import optuna

    indir = Path(args.indir)
    if not indir.exists():
        raise SystemExit(f"--indir does not exist: {indir}")

    # Load all compiled artifacts
    dist_items: List[Dict[str, Any]] = []
    for data_path in sorted(indir.glob("*.data.npz")):
        blob = np.load(data_path, allow_pickle=False)
        name = str(blob["dist_name"])
        qasm_path = indir / f"{name}.stateprep.qasm"
        if not qasm_path.exists():
            raise SystemExit(f"Missing QASM for {name}: expected {qasm_path}")

        dist_items.append(
            {
                "name": name,
                "grid": blob["grid"].astype(np.float64),
                "probs": blob["probs"].astype(np.float64),
                "alpha": float(blob["alpha"]),
                "ref_var": float(blob["ref_var"]),
                "ref_idx": int(blob["ref_idx"]),
                "num_qubits": int(blob["num_qubits"]),
                "qasm_path": qasm_path,
            }
        )

    if not dist_items:
        raise SystemExit(f"No *.data.npz found in {indir}")

    # Shared sampler (reuse across all calls)
    sampler_v2 = _make_sampler_v2(
        device=args.device,
        method=args.method,
        seed=args.seed,
        default_shots=args.shots,
    )

    # Load all state-prep circuits once
    for dd in dist_items:
        dd["stateprep"] = _load_stateprep_qasm(dd["qasm_path"])

    alpha_target = float(args.alpha)
    max_steps = int(args.max_steps)

    def objective(trial: optuna.Trial) -> float:
        # Algorithmic-only params to sweep
        epsilon = trial.suggest_float("epsilon", 0.01, 0.15)
        alpha_fail = trial.suggest_float("alpha_fail", 0.001, 0.05, log=True)
        prob_tol_mult = trial.suggest_float("prob_tol_mult", 0.05, 0.3)

        prob_tol = alpha_target * prob_tol_mult

        total_cost = 0.0
        total_err = 0.0
        max_err = 0.0

        for i, dd in enumerate(dist_items):
            var_hat, _, cost = solve_var_bisect_quantum(
                sampler_v2=sampler_v2,
                stateprep_asset_only=dd["stateprep"],
                grid_points=dd["grid"],
                probs=dd["probs"],
                alpha_target=alpha_target,
                epsilon=epsilon,
                alpha_fail=alpha_fail,
                prob_tol=prob_tol,
                max_steps=max_steps,
            )

            err = abs(var_hat - dd["ref_var"]) / (abs(dd["ref_var"]) + 1e-9)
            print(f"  [{dd['name']}] var_hat={var_hat:.4f}, ref={dd['ref_var']:.4f}, err={err:.6f}, cost={cost}")

            total_cost += float(cost)
            total_err += float(err)
            max_err = max(max_err, float(err))

            trial.report(total_cost / (i + 1), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg_cost = total_cost / len(dist_items)
        avg_err = total_err / len(dist_items)
        # same style you used: cost + strong penalty on error
        return avg_cost + 1e6 * avg_err + 1e5 * max_err

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.trials // 4))
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=int(args.trials), n_jobs=int(args.jobs))

    best = {
        "mode": "quantum_algo_only_qasm_run",
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "num_distributions": len(dist_items),
        "distributions": [d["name"] for d in dist_items],
        "alpha": alpha_target,
        "num_qubits": int(dist_items[0]["num_qubits"]),
        "aer": {
            "device": args.device,
            "method": args.method,
            "shots": int(args.shots),
        },
        "max_steps": int(args.max_steps),
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"[run] wrote {out_path}")


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    apc = sub.add_parser("compile", help="Compile state-prep circuits to OpenQASM via Classiq.")
    apc.add_argument("--outdir", type=str, required=True)
    apc.add_argument("--num-qubits", type=int, default=7)
    apc.add_argument("--alpha", type=float, default=0.05)
    apc.add_argument("--seed", type=int, default=0)
    apc.add_argument("--login", action="store_true", help="Call classiq.authenticate()")
    apc.add_argument("--debug-mode", action="store_true", help="Classiq debug_mode=True (slower, more structure)")
    apc.add_argument("--qasm3", action="store_true", help="Also request OpenQASM 3 output (in addition to 2)")
    apc.set_defaults(func=cmd_compile)

    apr = sub.add_parser("run", help="Run algorithmic-only sweep using QASM + Qiskit (no Classiq).")
    apr.add_argument("--indir", type=str, required=True)
    apr.add_argument("--trials", type=int, default=40)
    apr.add_argument("--jobs", type=int, default=1, help="Optuna parallel jobs (careful on single GPU).")
    apr.add_argument("--seed", type=int, default=0)
    apr.add_argument("--alpha", type=float, default=0.05)
    apr.add_argument("--max-steps", type=int, default=64)

    # Aer options (GPU node)
    apr.add_argument("--device", type=str, default="GPU", choices=["CPU", "GPU"])
    apr.add_argument("--method", type=str, default="statevector", help="Aer simulation method (e.g., statevector)")
    apr.add_argument("--shots", type=int, default=20000, help="Default shots for SamplerV2")

    apr.add_argument("--out", type=str, default="best_algo_params.json")
    apr.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
