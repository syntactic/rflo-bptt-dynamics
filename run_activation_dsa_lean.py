"""Activation-space DSA over trained runs, with a memory-lean loader.

Same computation as run_activation_dsa.py, but it loads one artifact at a time and
keeps only H and direction_idx. Artifacts carrying a full per-step weight history run
~1 GB each, and holding fifteen of them at once exhausts memory on a 26 GB machine.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

import analysis


def load_H_run(results_dir, effector, seed, rule):
    """Load only the fields DSA needs (H, direction_idx), then drop the rest."""
    path = Path(results_dir) / f"{effector}_seed{seed}_{rule}.pt"
    data = torch.load(path, map_location="cpu")
    lean = {"H": data["H"], "direction_idx": data["direction_idx"]}
    del data
    gc.collect()
    return lean


def main():
    p = argparse.ArgumentParser(description="Memory-lean activation DSA.")
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

    # Inference is NOT done here: BPTT-seedN and RFLO-seedN share init + target stream, so
    # a free label permutation overstates significance. Run the paired/block permutation on
    # the saved matrix instead (run_paired_permutation.py, which reads the labels sidecar).

    fig = analysis.plot_dsa_heatmap(
        similarities, labels, title=f"Activation DSA: RFLO vs BPTT ({args.effector})"
    )
    save_path = Path(args.results_dir) / f"{args.effector}_activation_dsa_heatmap.png"
    fig.savefig(save_path)
    print(f"Saved heatmap figure to: {save_path}")
    out = Path(args.results_dir) / f"{args.effector}_activation_dsa_matrix.npy"
    np.save(out, similarities)
    # Row-order sidecar so the paired-permutation runner pairs rows without re-deriving them.
    labels_out = out.with_name(out.name[: -len("_matrix.npy")] + "_labels.json")
    labels_out.write_text(json.dumps(labels))
    print(f"Saved DSA matrix to: {out}")
    print(f"Saved row-order labels to: {labels_out}")
    print(
        "Next: run_paired_permutation.py on this matrix for the correct paired inference."
    )


if __name__ == "__main__":
    main()
