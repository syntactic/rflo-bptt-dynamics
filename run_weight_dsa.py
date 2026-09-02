import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    import analysis
    import plot

    return Path, analysis, mo, np, plot, plt, sys, torch


@app.cell
def _(mo, sys):
    _default_effector = "RigidTendonArm26"
    if len(sys.argv) > 1 and sys.argv[-1] in ["ReluPointMass24", "RigidTendonArm26"]:
        _default_effector = sys.argv[-1]

    effector_selector = mo.ui.dropdown(
        options=["ReluPointMass24", "RigidTendonArm26"],
        value=_default_effector,
        label="Select Effector:",
    )
    return (effector_selector,)


@app.cell
def _(effector_selector):
    effector_selector


@app.cell
def _(Path):
    RESULTS_DIR = Path("results")
    SEEDS = range(5)
    RULES = ("BPTT", "RFLO")
    return RESULTS_DIR, RULES, SEEDS


@app.cell
def _(RESULTS_DIR, RULES, SEEDS, analysis, effector_selector):
    _effector = effector_selector.value
    runs = analysis.load_all_runs(RESULTS_DIR, _effector, SEEDS, RULES)
    pc_trajectories, pca = analysis.process_weight_trajectories(
        runs, bin_size=10, n_components=0.95
    )
    return pca, pc_trajectories, runs


@app.cell
def _(effector_selector, pca, pc_trajectories, plot, plt):
    _fig = plot.plot_weight_pca(
        pc_trajectories,
        pca=pca,
        title=f"Weight Trajectories (W_rec): BPTT vs RFLO ({effector_selector.value})",
    )
    _fig.savefig(f"results/{effector_selector.value}_weights_pca.png")
    plt.show()


@app.cell
def _(RESULTS_DIR, RULES, SEEDS, analysis, effector_selector, pc_trajectories):
    systems = []
    labels = []
    for _rule in RULES:
        for _seed in SEEDS:
            systems.append(pc_trajectories[(_rule, _seed)])
            labels.append(f"{_rule}-{_seed}")
    similarities = analysis.run_dsa(systems, n_delays=10, rank=10)
    summary = analysis.summarize_within_between(similarities, labels)
    p_val = analysis.calculate_p_value_of_dsa_distance(similarities, labels, RULES)

    for pair, dist in summary.items():
        print(f"{pair}: mean DSA distance = {dist:.4f}")
    print(f"Permutation p-value: {p_val:.5f}")

    _effector = effector_selector.value
    _fig = analysis.plot_dsa_heatmap(
        similarities, labels, title=f"Weight-Trajectory DSA ({_effector})"
    )
    _fig.savefig(RESULTS_DIR / f"{_effector}_weight_dsa_heatmap.png")


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
