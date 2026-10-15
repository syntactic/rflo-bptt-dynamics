"""Activation DSA between BPTT and RFLO runs in one results directory.

Artifacts are loaded one at a time and only H and direction_idx are kept. With weight
history an arm artifact is 0.67 GB at 40k steps, and fifteen at once ran my 24 GB
machine out of memory.

Saves the distance matrix, a row-order label file and a heatmap. It prints no p-value;
run run_paired_permutation.py on the saved matrix for that.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

import analysis


def load_H_run(results_dir, effector, seed, rule):
    """Load one artifact and keep only H and direction_idx."""
    path = Path(results_dir) / f"{effector}_seed{seed}_{rule}.pt"
    data = torch.load(path, map_location="cpu")
    lean = {"H": data["H"], "direction_idx": data["direction_idx"]}
    del data
    gc.collect()
    return lean


def main():
    p = argparse.ArgumentParser(description="Activation DSA by learning rule.")
    p.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"])
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--n-delays", type=int, default=10)
    p.add_argument("--rank", type=int, default=20)
    args = p.parse_args()

    rules = ("BPTT", "RFLO")
    systems, labels = [], []
    for rule in rules:
        for seed in args.seeds:
            run = load_H_run(args.results_dir, args.effector, seed, rule)
            systems.append(analysis.extract_per_direction_trajectories(run))
            labels.append(f"{rule}-{seed}")
            del run
            gc.collect()

    print(
        f"Running Activation DSA on {len(systems)} systems "
        f"(n_delays={args.n_delays}, rank={args.rank})..."
    )
    similarities = analysis.run_dsa(systems, n_delays=args.n_delays, rank=args.rank)

    summary = analysis.summarize_within_between(similarities, labels)
    for pair, mean_dist in summary.items():
        print(f"{pair}: mean DSA distance = {mean_dist:.4f}")

    # BPTT and RFLO runs of one seed share an initialization and target stream, so a free
    # label permutation overstates significance. The paired test runs separately.

    fig = analysis.plot_dsa_heatmap(
        similarities, labels, title=f"Activation DSA: RFLO vs BPTT ({args.effector})"
    )
    save_path = Path(args.results_dir) / f"{args.effector}_activation_dsa_heatmap.png"
    fig.savefig(save_path)
    print(f"Saved heatmap figure to: {save_path}")
    out = Path(args.results_dir) / f"{args.effector}_activation_dsa_matrix.npy"
    np.save(out, similarities)
    # run_paired_permutation.py reads the row order from this file.
    labels_out = out.with_name(out.name[: -len("_matrix.npy")] + "_labels.json")
    labels_out.write_text(json.dumps(labels))
    print(f"Saved DSA matrix to: {out}")
    print(f"Saved row-order labels to: {labels_out}")
    print("Next: run_paired_permutation.py on this matrix for the paired p-value.")


if __name__ == "__main__":
    main()
