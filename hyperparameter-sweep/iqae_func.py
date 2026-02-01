from pathlib import Path
import csv
import argparse
import numpy as np
from tqdm import tqdm
from itertools import product

from qae_sweep import (
    _make_sampler_v2,
    _load_stateprep_qasm,
    _estimate_tail_prob_iae,
)


def run_classical_mc(probs, threshold_idx, n_samples, seed):
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=n_samples, p=probs)
    return np.mean(samples < threshold_idx)


def get_threshold_idx(probs, alpha):
    """Get threshold index for given alpha (VaR level)."""
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, alpha, side="left"))


def run_all_dists(indir, outfile, seed=42):
    n_samples_list = np.unique(np.logspace(1, 7, 100).astype(int))
    epsilons = np.clip(np.logspace(-4, -0.3, 80), 1e-6, 0.49)
    alpha_fails = [0.01, 0.05, 0.1]  # Confidence: 99%, 95%, 90%
    var_alphas = [0.01, 0.05, 0.10]  # VaR levels: 1%, 5%, 10%

    # Load circuits
    precompiled = {}
    for data_path in sorted(indir.glob("*.data.npz")):
        blob = np.load(data_path)
        dist_name = str(blob["dist_name"])
        probs = blob["probs"].astype(np.float64)
        num_qubits = int(blob["num_qubits"])
        stateprep = _load_stateprep_qasm(indir / f"{dist_name}.stateprep.qasm")
        precompiled[dist_name] = (probs, stateprep, num_qubits)
        print(f"{dist_name}: loaded")

    sampler = _make_sampler_v2("GPU", "statevector", seed, 1024)

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dist", "var_alpha", "queries", "epsilon", "alpha_fail", "p_hat", "true_p", "error"])

        # Classical - vary var_alpha
        rng = np.random.default_rng(seed)
        for dist_name, (probs, _, _) in tqdm(precompiled.items(), desc="Classical"):
            for var_alpha in var_alphas:
                ref_idx = get_threshold_idx(probs, var_alpha)
                true_p = float(np.sum(probs[:ref_idx]))
                for n in n_samples_list:
                    samples = rng.choice(len(probs), size=n, p=probs)
                    p_hat = np.mean(samples < ref_idx)
                    writer.writerow(["classical", dist_name, var_alpha, n, "", "", p_hat, true_p, abs(p_hat - true_p)])
            f.flush()

        # Quantum - vary epsilon, alpha_fail, var_alpha
        total = len(precompiled) * len(var_alphas) * len(epsilons) * len(alpha_fails)
        pbar = tqdm(total=total, desc="Quantum")

        for dist_name, (probs, stateprep, num_qubits) in precompiled.items():
            for var_alpha in var_alphas:
                ref_idx = get_threshold_idx(probs, var_alpha)
                true_p = float(np.sum(probs[:ref_idx]))

                for eps, af in product(epsilons, alpha_fails):
                    est = _estimate_tail_prob_iae(
                        sampler_v2=sampler,
                        stateprep_asset_only=stateprep,
                        num_asset_qubits=num_qubits,
                        threshold_index=ref_idx,
                        epsilon=eps,
                        alpha_fail=af,
                    )
                    writer.writerow(
                        ["quantum", dist_name, var_alpha, est.cost_oracle_queries, eps, af, est.p_hat, true_p,
                         abs(est.p_hat - true_p)])
                    pbar.update(1)
                f.flush()
        pbar.close()

    print(f"Done: {outfile}")


def run_single_dist(indir, outfile, dist_name="normal", seed=42):
    n_samples_list = np.unique(np.logspace(1, 7, 0).astype(int))
    epsilons = np.clip(np.logspace(-2.5, -0.5, 50), 1e-4, 0.1)
    alpha_fails = [0.05]
    var_alphas = [0.05, 0.10]

    data_path = indir / f"{dist_name}.data.npz"
    blob = np.load(data_path)
    probs = blob["probs"].astype(np.float64)
    num_qubits = int(blob["num_qubits"])
    stateprep = _load_stateprep_qasm(indir / f"{dist_name}.stateprep.qasm")
    print(f"{dist_name}: num_qubits={num_qubits}")

    sampler = _make_sampler_v2("GPU", "statevector", seed, 1024)

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dist", "var_alpha", "queries", "epsilon", "alpha_fail", "p_hat", "true_p", "error"])

        # Classical
        rng = np.random.default_rng(seed)
        for var_alpha in tqdm(var_alphas, desc="Classical alphas"):
            ref_idx = get_threshold_idx(probs, var_alpha)
            true_p = float(np.sum(probs[:ref_idx]))
            for n in n_samples_list:
                samples = rng.choice(len(probs), size=n, p=probs)
                p_hat = np.mean(samples < ref_idx)
                writer.writerow(["classical", dist_name, var_alpha, n, "", "", p_hat, true_p, abs(p_hat - true_p)])
        f.flush()

        # Quantum
        total = len(var_alphas) * len(epsilons) * len(alpha_fails)
        pbar = tqdm(total=total, desc="Quantum")

        for var_alpha in var_alphas:
            ref_idx = get_threshold_idx(probs, var_alpha)
            true_p = float(np.sum(probs[:ref_idx]))

            for eps, af in product(epsilons, alpha_fails):
                est = _estimate_tail_prob_iae(
                    sampler_v2=sampler,
                    stateprep_asset_only=stateprep,
                    num_asset_qubits=num_qubits,
                    threshold_index=ref_idx,
                    epsilon=eps,
                    alpha_fail=af,
                )
                writer.writerow(["quantum", dist_name, var_alpha, est.cost_oracle_queries, eps, af, est.p_hat, true_p,
                                 abs(est.p_hat - true_p)])
                pbar.update(1)
            f.flush()
        pbar.close()

    print(f"Done: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "single"], required=True)
    parser.add_argument("--indir", default="build_qasm_10")
    parser.add_argument("--dist", default="normal")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    indir = Path(args.indir)

    if args.mode == "all":
        run_all_dists(indir, "scaling_all_dists.csv", args.seed)
    else:
        run_single_dist(indir, f"scaling_{args.dist}_detailed.csv", args.dist, args.seed)