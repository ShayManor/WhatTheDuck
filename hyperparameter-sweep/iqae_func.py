from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm

from qae_sweep import (
    _make_sampler_v2,
    _load_stateprep_qasm,
    _estimate_tail_prob_iae,
)

if __name__ == '__main__':
    indir = Path("build_qasm")
    seed = 42

    n_samples_list = np.unique(np.logspace(1, 6, 100).astype(int))  # 100 points
    epsilons = np.clip(np.logspace(-2.5, -0.3, 50), a_min=1e-6, a_max=0.49)

    # Load
    precompiled = {}
    for data_path in sorted(indir.glob("*.data.npz")):
        blob = np.load(data_path)
        dist_name = str(blob["dist_name"])
        probs = blob["probs"].astype(np.float64)
        num_qubits = int(blob["num_qubits"])
        ref_idx = int(blob["ref_idx"])
        true_p = float(np.sum(probs[:ref_idx]))
        stateprep = _load_stateprep_qasm(indir / f"{dist_name}.stateprep.qasm")
        precompiled[dist_name] = (probs, stateprep, num_qubits, ref_idx, true_p)
        print(f"{dist_name}: ref_idx={ref_idx}, true_p={true_p:.6f}")

    sampler = _make_sampler_v2("GPU", "statevector", seed, 1024)

    with open("scaling_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dist", "queries", "epsilon", "p_hat", "true_p", "error"])

        # Classical
        rng = np.random.default_rng(seed)
        for dist_name, (probs, _, _, ref_idx, true_p) in tqdm(precompiled.items(), desc="Dists (classical)"):
            for n in tqdm(n_samples_list, desc=f"{dist_name}", leave=False):
                samples = rng.choice(len(probs), size=n, p=probs)
                p_hat = np.mean(samples < ref_idx)
                writer.writerow(["classical", dist_name, n, "", p_hat, true_p, abs(p_hat - true_p)])
            f.flush()

        # Quantum
        for dist_name, (probs, stateprep, num_qubits, ref_idx, true_p) in tqdm(precompiled.items(), desc="Dists (quantum)"):
            for eps in tqdm(epsilons, desc=f"{dist_name}", leave=False):
                est = _estimate_tail_prob_iae(
                    sampler_v2=sampler,
                    stateprep_asset_only=stateprep,
                    num_asset_qubits=num_qubits,
                    threshold_index=ref_idx,
                    epsilon=eps,
                    alpha_fail=0.05,
                )
                writer.writerow(["quantum", dist_name, est.cost_oracle_queries, eps, est.p_hat, true_p, abs(est.p_hat - true_p)])
            f.flush()

    print("Done")