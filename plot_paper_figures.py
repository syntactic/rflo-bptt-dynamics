"""Publication figure generation script for RFLO vs BPTT motor learning paper."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analysis
import plot


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Paper Figure Generator",
        description="Generate all publication-ready standalone figures for BPTT vs RFLO dynamics",
    )
    parser.add_argument(
        "effector",
        choices=["ReluPointMass24", "RigidTendonArm26"],
        help="MotorNet effector model",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of seeds (e.g. --seeds 0 1 2 3 4)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds if --seeds is omitted (default: 5)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing .pt artifacts (default: 'results')",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: 'png')",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for raster outputs (default: 300)"
    )
    return parser.parse_args()


def generate_all_figures(
    effector: str,
    seeds: list[int],
    results_dir: Path,
    img_format: str = "png",
    dpi: int = 300,
):
    rules = ("BPTT", "RFLO")
    print(f"Loading runs for {effector} across seeds {seeds} from {results_dir}...")
    runs = analysis.load_all_runs(results_dir, effector, seeds, rules)

    # 1. Behavioral Comparison & Training Curves
    print("Generating Behavioral & Learning Curve Summary...")
    fig_beh, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Top Row: Reach Trajectories for seed 0 (or first seed)
    first_seed = seeds[0]
    bptt_run = runs[("BPTT", first_seed)]
    rflo_run = runs[("RFLO", first_seed)]

    # Artifacts written before 2026-09-12 scored success against an unscaled 0.05,
    # which is 50% of the arm's reach and 10% of the point mass's.
    radius = bptt_run.get("success_radius")
    if radius is None:
        radius = 0.05
        print("WARNING: no success_radius in artifact; legacy unscaled 0.05 assumed.")
    frac = radius / bptt_run.get("reaching_distance", analysis.reach_distance(effector))

    plot.plot_reach_trajectories(
        bptt_run["FT"],
        bptt_run["goal"],
        bptt_run["direction_idx"],
        ax=axes[0, 0],
        title=f"BPTT Reaches (Seed {first_seed})",
    )
    plot.plot_reach_trajectories(
        rflo_run["FT"],
        rflo_run["goal"],
        rflo_run["direction_idx"],
        ax=axes[0, 1],
        title=f"RFLO Reaches (Seed {first_seed})",
    )

    # Top Right: Per-direction comparison for first seed
    per_dir_data = {
        "BPTT": bptt_run["metrics"][-1][1],
        "RFLO": rflo_run["metrics"][-1][1],
    }
    directions = sorted(set().union(*(d.keys() for d in per_dir_data.values())))
    x = np.arange(len(directions))
    width = 0.35
    axes[0, 2].bar(
        x - width / 2,
        [per_dir_data["BPTT"][d][0] for d in directions],
        width,
        label="BPTT",
        color="tab:blue",
    )
    axes[0, 2].bar(
        x + width / 2,
        [per_dir_data["RFLO"][d][0] for d in directions],
        width,
        label="RFLO",
        color="tab:orange",
    )
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(directions)
    axes[0, 2].set_xlabel("Target Direction Index")
    axes[0, 2].set_ylabel(f"Final Success Rate ({frac:.0%} of reach)")
    axes[0, 2].set_title(f"Directional Success (Seed {first_seed})")
    axes[0, 2].legend()

    # Bottom Row: Training Curves averaged across all seeds
    bptt_losses = np.mean([runs[("BPTT", s)]["losses"] for s in seeds], axis=0)
    rflo_losses = np.mean([runs[("RFLO", s)]["losses"] for s in seeds], axis=0)
    axes[1, 0].plot(bptt_losses, label="BPTT", color="tab:blue", lw=1.5)
    axes[1, 0].plot(rflo_losses, label="RFLO", color="tab:orange", lw=1.5)
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Position Loss (L1)")
    axes[1, 0].set_title("Mean Training Loss")
    axes[1, 0].legend()

    # Success rate and Path deviation over training steps (align by common evaluation steps)
    all_runs_metrics = [
        runs[(r, s)]["metrics"] for r in ("BPTT", "RFLO") for s in seeds
    ]
    common_steps = sorted(
        set.intersection(*(set(step for step, _ in m) for m in all_runs_metrics))
    )

    def _extract_metric_curve(rule, metric_idx):
        per_seed_curves = []
        for s in seeds:
            step_to_val = {
                step: np.mean([v[metric_idx] for v in per_dir.values()])
                for step, per_dir in runs[(rule, s)]["metrics"]
            }
            per_seed_curves.append([step_to_val[step] for step in common_steps])
        return np.mean(per_seed_curves, axis=0)

    bptt_succ = _extract_metric_curve("BPTT", 0)
    rflo_succ = _extract_metric_curve("RFLO", 0)
    axes[1, 1].plot(
        common_steps, bptt_succ, marker="o", ms=4, label="BPTT", color="tab:blue"
    )
    axes[1, 1].plot(
        common_steps, rflo_succ, marker="o", ms=4, label="RFLO", color="tab:orange"
    )
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Mean Success Rate")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title(f"Success Rate (within {frac:.0%} of reach)")
    axes[1, 1].legend()

    units = "m" if "Arm" in effector else "a.u."
    bptt_dev = _extract_metric_curve("BPTT", 1)
    rflo_dev = _extract_metric_curve("RFLO", 1)
    axes[1, 2].plot(
        common_steps, bptt_dev, marker="o", ms=4, label="BPTT", color="tab:blue"
    )
    axes[1, 2].plot(
        common_steps, rflo_dev, marker="o", ms=4, label="RFLO", color="tab:orange"
    )
    axes[1, 2].set_xlabel("Training Step")
    axes[1, 2].set_ylabel(f"Mean Path Deviation ({units})")
    axes[1, 2].set_title("Straight-Line Path Deviation")
    axes[1, 2].legend()

    fig_beh.tight_layout()
    beh_save = results_dir / f"{effector}_behavior_summary.{img_format}"
    fig_beh.savefig(beh_save, dpi=dpi)
    print(f"Saved: {beh_save}")
    plt.close(fig_beh)

    # 2. Activation State-Space PCA
    print("Generating Activation State-Space PCA Figure...")
    pc_act_trajs, pca_act = analysis.process_activation_trajectories(
        runs, n_components=0.95
    )
    fig_act_pca, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot.plot_activation_pca(
        pc_act_trajs[("BPTT", first_seed)],
        pca=pca_act,
        title=f"BPTT Hidden Activations (Seed {first_seed})",
        ax=ax1,
    )
    plot.plot_activation_pca(
        pc_act_trajs[("RFLO", first_seed)],
        pca=pca_act,
        title=f"RFLO Hidden Activations (Seed {first_seed})",
        ax=ax2,
    )
    fig_act_pca.tight_layout()
    act_pca_save = results_dir / f"{effector}_activation_pca.{img_format}"
    fig_act_pca.savefig(act_pca_save, dpi=dpi)
    print(f"Saved: {act_pca_save}")
    plt.close(fig_act_pca)

    # 3. Activation DSA Heatmap
    print("Generating Activation DSA Heatmap...")
    act_systems, act_labels = [], []
    for rule in rules:
        for seed in seeds:
            trials = analysis.extract_per_direction_trajectories(runs[(rule, seed)])
            act_systems.append(trials)
            act_labels.append(f"{rule}-{seed}")
    act_sims = analysis.run_dsa(act_systems, n_delays=10, rank=20)
    fig_act_dsa = analysis.plot_dsa_heatmap(
        act_sims, act_labels, title=f"Activation DSA: RFLO vs BPTT ({effector})"
    )
    act_dsa_save = results_dir / f"{effector}_activation_dsa_heatmap.{img_format}"
    fig_act_dsa.savefig(act_dsa_save, dpi=dpi)
    print(f"Saved: {act_dsa_save}")
    plt.close(fig_act_dsa)

    # 4. Weight PCA Trajectories
    print("Generating Weight PCA Trajectory Figure...")
    pc_weight_trajs, pca_weight = analysis.process_weight_trajectories(
        runs, bin_size=10, n_components=0.95
    )
    fig_weight_pca = plot.plot_weight_pca(
        pc_weight_trajs,
        pca=pca_weight,
        title=f"Recurrent Weight Learning Trajectories ({effector})",
    )
    weight_pca_save = results_dir / f"{effector}_weights_pca.{img_format}"
    fig_weight_pca.savefig(weight_pca_save, dpi=dpi)
    print(f"Saved: {weight_pca_save}")
    plt.close(fig_weight_pca)

    print("\nAll publication figures successfully generated!")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    generate_all_figures(
        args.effector, seeds, results_dir, img_format=args.format, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
