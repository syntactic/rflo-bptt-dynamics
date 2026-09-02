"""Headless CLI runner for weight trajectory PCA and DSA analysis."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import analysis
import plot


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Weight DSA Runner",
        description="Compute PCA and Dynamical Similarity Analysis (DSA) on recurrent weight trajectories",
    )
    parser.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"], help="MotorNet effector model")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit list of seeds (e.g. --seeds 0 1 2 3 4)")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds if --seeds is omitted (default: 5)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing .pt artifacts (default: 'results')")
    parser.add_argument("--bin-size", type=int, default=10, help="Coarse-graining bin size for weight trajectory smoothing (default: 10)")
    parser.add_argument("--n-delays", type=int, default=10, help="Hankel delay embedding size (default: 10)")
    parser.add_argument("--rank", type=int, default=10, help="SVD truncation rank for weight DSA (default: 10)")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib window")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    rules = ("BPTT", "RFLO")

    print(f"Loading runs for {args.effector} across seeds {seeds} from {results_dir}...")
    runs = analysis.load_all_runs(results_dir, args.effector, seeds, rules)

    print(f"Processing weight trajectories (bin_size={args.bin_size})...")
    pc_trajectories, pca = analysis.process_weight_trajectories(runs, bin_size=args.bin_size, n_components=0.95)

    # 1. Plot & Save PCA Trajectory Figure
    fig_pca = plot.plot_weight_pca(
        pc_trajectories,
        pca=pca,
        title=f"Weight Trajectories (W_rec): BPTT vs RFLO ({args.effector})",
    )
    pca_save_path = results_dir / f"{args.effector}_weights_pca.png"
    fig_pca.savefig(pca_save_path)
    print(f"Saved weight PCA figure to: {pca_save_path}")

    # 2. Compute DSA on PC trajectories
    systems, labels = [], []
    for rule in rules:
        for seed in seeds:
            systems.append(pc_trajectories[(rule, seed)])
            labels.append(f"{rule}-{seed}")

    print(f"Running Weight DSA on {len(systems)} systems (n_delays={args.n_delays}, rank={args.rank})...")
    similarities = analysis.run_dsa(systems, n_delays=args.n_delays, rank=args.rank)

    summary = analysis.summarize_within_between(similarities, labels)
    for pair, dist in summary.items():
        print(f"{pair}: mean DSA distance = {dist:.4f}")

    if len(similarities) % 2 == 0 and len(similarities) >= 4:
        p_val = analysis.calculate_p_value_of_dsa_distance(similarities, labels, rules)
        print(f"Exact label-permutation p-value: {p_val:.5f}")

    fig_dsa = analysis.plot_dsa_heatmap(
        similarities, labels, title=f"Weight-Trajectory DSA ({args.effector})"
    )
    dsa_save_path = results_dir / f"{args.effector}_weight_dsa_heatmap.png"
    fig_dsa.savefig(dsa_save_path)
    print(f"Saved weight DSA heatmap to: {dsa_save_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig_pca)
    plt.close(fig_dsa)


if __name__ == "__main__":
    main()
