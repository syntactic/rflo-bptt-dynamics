import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import numpy as np
    from pathlib import Path
    from DSA import coarse_grain, pca_reduce

    return Path, coarse_grain, np, pca_reduce, torch


@app.cell
def _(Path):
    RESULTS_DIR = Path("results")
    SEEDS = range(5)
    RULES = ["BPTT", "RFLO"]
    return RESULTS_DIR, RULES, SEEDS


@app.cell
def _(RESULTS_DIR, RULES, SEEDS, torch):
    centered_trajectories = {}
    for _rule in RULES:
        for _seed in SEEDS:
            run = torch.load(RESULTS_DIR / f"seed{_seed}_{_rule}.pt")
            W_traj = torch.stack(run["weight_history"]).detach().cpu().numpy()
            centered_trajectories[(_rule, _seed)] = W_traj - W_traj[0:1, :]
    return (centered_trajectories,)


@app.cell
def _(centered_trajectories, coarse_grain):
    smoothed_trajectories = {}
    for _key, _traj in centered_trajectories.items():
        smoothed_trajectories[_key] = coarse_grain(_traj, bin_size=10) # shape (800, 4096)
    return (smoothed_trajectories,)


@app.cell
def _(np, pca_reduce, smoothed_trajectories):
    # shape (800 * 10, 4096)
    all_data = np.concatenate([smoothed_trajectories[k] for k in smoothed_trajectories], axis=0)

    reduced_all, pca = pca_reduce(all_data, n_components=0.95, return_pca=True, verbose=True)
    return (pca,)


@app.cell
def _(pca, smoothed_trajectories):
    pc_trajectories = {}
    for _key, _traj in smoothed_trajectories.items():
        pc_trajectories[_key] = pca.transform(_traj) # shape (800, 11 components)
    return (pc_trajectories,)


@app.cell
def _(pc_trajectories, pca):
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(figsize=(7, 6))

    for (_rule, _seed), _traj in pc_trajectories.items():
        color = "tab:blue" if _rule == "BPTT" else "tab:orange"
        alpha = 0.8
        label = _rule if _seed == 0 else None  # Avoid duplicate legend entries

        # Plot trajectory line
        ax.plot(_traj[:, 0], _traj[:, 1], color=color, alpha=alpha, label=label)
        # Start point (should coincide for matching seeds!)
        ax.scatter(_traj[0, 0], _traj[0, 1], color="black", s=20, zorder=5)
        # End point
        ax.scatter(_traj[-1, 0], _traj[-1, 1], color=color, marker="x", s=50, zorder=5)

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Weight Trajectories (W_rec): BPTT vs RFLO")
    ax.legend()
    plt.show()
    return


@app.cell
def _(RULES, SEEDS, pc_trajectories):
    import analysis
    import run_dsa
    systems = []
    labels = []
    for _rule in RULES:
        for _seed in SEEDS:
            systems.append(pc_trajectories[(_rule, _seed)])
            labels.append(f"{_rule}-{_seed}")
    similarities = analysis.run_dsa(systems, n_delays=10, rank=10)
    summary = run_dsa.summarize_within_between(similarities, labels)
    p_val = run_dsa.calculate_p_value_of_dsa_distance(similarities, labels, RULES)

    for pair, dist in summary.items():
        print(f"{pair}: mean DSA distance = {dist:.4f}")
    print(f"Permutation p-value: {p_val:.5f}")

    fig = analysis.plot_dsa_heatmap(similarities, labels, title="Weight-Trajectory DSA (W_rec)")
    fig.savefig("results/weight_dsa_heatmap.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
