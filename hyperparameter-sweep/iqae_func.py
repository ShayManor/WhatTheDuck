from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import time

import numpy as np

from qae_sweep import (
    _make_sampler_v2,
    _load_stateprep_qasm,
    compute_true_var,
    _estimate_tail_prob_iae,
)


def run_classical_mc(probs, threshold_idx, n_samples, seed):
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=n_samples, p=probs)
    p_hat = np.mean(samples < threshold_idx)
    return p_hat


def run_quantum_ae(sampler, stateprep, num_qubits, threshold_idx, epsilon, alpha_fail=0.05):
    est = _estimate_tail_prob_iae(
        sampler_v2=sampler,
        stateprep_asset_only=stateprep,
        num_asset_qubits=num_qubits,
        threshold_index=threshold_idx,
        epsilon=epsilon,
        alpha_fail=alpha_fail,
    )
    return est.p_hat, est.cost_oracle_queries


if __name__ == '__main__':
    indir = Path("build_qasm")
    alpha = 0.05
    seed = 42

    n_samples_list = np.unique(np.logspace(1, 7, 500).astype(int))
    epsilons = np.logspace(-3.5, -0.3, 150)

    # Load precompiled circuits
    print("Loading precompiled circuits...")
    precompiled = {}
    for data_path in sorted(indir.glob("*.data.npz")):
        blob = np.load(data_path)
        dist_name = str(blob["dist_name"])
        probs = blob["probs"].astype(np.float64)
        grid = blob["grid"].astype(np.float64)
        num_qubits = int(blob["num_qubits"])
        ref_idx = int(blob["ref_idx"])
        true_p = float(np.sum(probs[:ref_idx]))

        qasm_path = indir / f"{dist_name}.stateprep.qasm"
        stateprep = _load_stateprep_qasm(qasm_path)

        precompiled[dist_name] = {
            "probs": probs,
            "grid": grid,
            "stateprep": stateprep,
            "num_qubits": num_qubits,
            "ref_idx": ref_idx,
            "true_p": true_p,
        }
        print(f"  {dist_name}: ref_idx={ref_idx}, true_p={true_p:.6f}")

    results = []

    # CLASSICAL
    print(f"\nClassical MC: {len(n_samples_list) * len(precompiled)} tasks...")
    t0 = time.time()

    def classical_task(dist_name, n_samples):
        d = precompiled[dist_name]
        p_hat = run_classical_mc(d["probs"], d["ref_idx"], n_samples, seed)
        return {
            "method": "classical",
            "dist": dist_name,
            "queries": int(n_samples),
            "epsilon": None,
            "p_hat": p_hat,
            "true_p": d["true_p"],
            "error": abs(p_hat - d["true_p"]),
        }

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(classical_task, d, n) for d in precompiled for n in n_samples_list]
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(futures)}")

    print(f"Classical: {time.time()-t0:.1f}s")

    # QUANTUM
    print(f"\nQuantum AE: {len(epsilons) * len(precompiled)} tasks...")
    t0 = time.time()

    def quantum_task(dist_name, eps):
        d = precompiled[dist_name]
        sampler = _make_sampler_v2("GPU", "statevector", seed, 1024)
        p_hat, queries = run_quantum_ae(sampler, d["stateprep"], d["num_qubits"], d["ref_idx"], eps)
        return {
            "method": "quantum",
            "dist": dist_name,
            "queries": int(queries),
            "epsilon": eps,
            "p_hat": p_hat,
            "true_p": d["true_p"],
            "error": abs(p_hat - d["true_p"]),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(quantum_task, d, e) for d in precompiled for e in epsilons]
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(futures)}")

    print(f"Quantum: {time.time()-t0:.1f}s")

    # CSV
    out_path = Path("scaling_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dist", "queries", "epsilon", "p_hat", "true_p", "error"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {out_path}")