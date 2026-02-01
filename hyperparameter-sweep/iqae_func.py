from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import time

import numpy as np
from tqdm import tqdm

from qae_sweep import (
    _make_sampler_v2,
    _load_stateprep_qasm,
    _estimate_tail_prob_iae,
)


def run_classical_mc(probs, threshold_idx, n_samples, seed):
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=n_samples, p=probs)
    return np.mean(samples < threshold_idx)


if __name__ == '__main__':
    indir = Path("build_qasm")
    seed = 42

    n_samples_list = np.unique(np.logspace(1, 7, 300).astype(int))
    epsilons = np.logspace(-3, -0.3, 80)

    # Load precompiled
    print("Loading circuits...")
    precompiled = {}
    for data_path in sorted(indir.glob("*.data.npz")):
        blob = np.load(data_path)
        dist_name = str(blob["dist_name"])
        probs = blob["probs"].astype(np.float64)
        num_qubits = int(blob["num_qubits"])
        ref_idx = int(blob["ref_idx"])
        true_p = float(np.sum(probs[:ref_idx]))
        stateprep = _load_stateprep_qasm(indir / f"{dist_name}.stateprep.qasm")
        precompiled[dist_name] = {
            "probs": probs,
            "stateprep": stateprep,
            "num_qubits": num_qubits,
            "ref_idx": ref_idx,
            "true_p": true_p,
        }
        print(f"  {dist_name}: ref_idx={ref_idx}, true_p={true_p:.6f}")

    results = []

    # CLASSICAL
    print(f"\nClassical: {len(n_samples_list) * len(precompiled)} tasks")
    t0 = time.time()


    def classical_task(args):
        dist_name, n_samples = args
        d = precompiled[dist_name]
        p_hat = run_classical_mc(d["probs"], d["ref_idx"], n_samples, seed + n_samples)
        return ("classical", dist_name, int(n_samples), None, p_hat, d["true_p"], abs(p_hat - d["true_p"]))


    classical_args = [(d, n) for d in precompiled for n in n_samples_list]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results.extend(tqdm(pool.map(classical_task, classical_args, chunksize=50),
                            total=len(classical_args), desc="Classical"))

    print(f"Classical: {time.time() - t0:.1f}s")

    # QUANTUM
    print(f"\nQuantum: {len(epsilons) * len(precompiled)} tasks")
    t0 = time.time()

    quantum_args = [(d, e) for d in precompiled for e in epsilons]


    def quantum_task(args):
        dist_name, eps = args
        d = precompiled[dist_name]
        sampler = _make_sampler_v2("GPU", "statevector", seed, 1024)
        est = _estimate_tail_prob_iae(
            sampler_v2=sampler,
            stateprep_asset_only=d["stateprep"],
            num_asset_qubits=d["num_qubits"],
            threshold_index=d["ref_idx"],
            epsilon=eps,
            alpha_fail=0.05,
        )
        return ("quantum", dist_name, int(est.cost_oracle_queries), eps, est.p_hat, d["true_p"],
                abs(est.p_hat - d["true_p"]))


    with ThreadPoolExecutor(max_workers=128) as pool:
        futures = [pool.submit(quantum_task, arg) for arg in quantum_args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Quantum", smoothing=0.1):
            results.append(f.result())

    print(f"Quantum: {time.time() - t0:.1f}s")

    # CSV
    with open("scaling_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dist", "queries", "epsilon", "p_hat", "true_p", "error"])
        writer.writerows(results)

    print(f"\nDone: {len(results)} rows")