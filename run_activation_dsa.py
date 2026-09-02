"""Activation-space DSA between BPTT and RFLO across seeds saved by run_experiment.py."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import analysis


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Activation DSA Runner",
        description="Compute Dynamical Similarity Analysis (DSA) on recurrent hidden activations",
    )
    parser.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"], help="MotorNet effector model")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit list of seeds (e.g. --seeds 0 1 2 3 4)")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds if --seeds is omitted (default: 5)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing .pt artifacts (default: 'results')")
    parser.add_argument("--n-delays", type=int, default=10, help="Hankel delay embedding size (default: 10)")
    parser.add_argument("--rank", type=int, default=20, help="SVD truncation rank for DSA (default: 20)")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib window")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    rules = ("BPTT", "RFLO")

    print(f"Loading runs for {args.effector} across seeds {seeds} from {results_dir}...")
    runs = analysis.load_all_runs(results_dir, args.effector, seeds, rules)

    systems, labels = [], []
    for rule in rules:
        for seed in seeds:
            trials = analysis.extract_per_direction_trajectories(runs[(rule, seed)])
            systems.append(trials)
            labels.append(f"{rule}-{seed}")

    print(f"Running Activation DSA on {len(systems)} systems (n_delays={args.n_delays}, rank={args.rank})...")
    similarities = analysis.run_dsa(systems, n_delays=args.n_delays, rank=args.rank)

    summary = analysis.summarize_within_between(similarities, labels)
    for pair, mean_dist in summary.items():
        print(f"{pair}: mean DSA distance = {mean_dist:.4f}")

    if len(similarities) % 2 == 0 and len(similarities) >= 4:
        p_val = analysis.calculate_p_value_of_dsa_distance(similarities, labels, rules)
        print(f"Exact label-permutation p-value: {p_val:.5f}")

    fig = analysis.plot_dsa_heatmap(
        similarities, labels, title=f"Activation DSA: RFLO vs BPTT ({args.effector})"
    )
    save_path = results_dir / f"{args.effector}_activation_dsa_heatmap.png"
    fig.savefig(save_path)
    print(f"Saved heatmap figure to: {save_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
