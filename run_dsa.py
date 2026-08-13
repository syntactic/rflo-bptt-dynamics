"""Phase 3: activation-space DSA between BPTT and RFLO, across all seeds saved
by run_experiment.py. Loads saved (seed, rule) artifacts only -- no live
trainer/env objects -- per the Phase 2.5/3 split in remaining_work_spec.md.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import analysis

RESULTS_DIR = Path("results")
SEEDS = range(5)
RULES = ("BPTT", "RFLO")


def load_run(seed, rule):
    return torch.load(RESULTS_DIR / f"seed{seed}_{rule}.pt")


def per_direction_trials(run):
    """Reshape one saved run's H -- (T, n_targets, n_rec) -- into the list of
    (T, n_rec) per-direction trajectories that DSA expects for a single
    system's multiple trials (see check_method()'s "self-pairwise" handling
    in DSA/dsa.py: nested lists = multiple samples of the same system).

    Assumes H's batch dim is ordered by direction, i.e. this run's final
    inference() call used options={"direction_idx": np.arange(n_targets)} --
    true for every run produced by run_experiment.py. Asserted explicitly
    since silently mismatched ordering would corrupt every downstream
    distance without erroring.
    """
    n_targets = run["H"].shape[1]
    assert np.array_equal(run["direction_idx"].squeeze(), np.arange(n_targets)), (
        "H's batch dim isn't ordered by direction -- was this run saved with "
        "direction_idx=np.arange(n_targets)?"
    )
    return list(run["H"].permute(1, 0, 2))


def summarize_within_between(similarities, labels):
    """Mean pairwise DSA distance grouped by (rule, rule) pair -- the numeric
    version of "do same-rule runs cluster together" that's easy to eyeball on
    a small heatmap but hard to judge reliably as it grows (10x10 already
    strains it). Groups on the rule prefix of each label (e.g. "BPTT-3" ->
    "BPTT"), so this generalizes past exactly two rules if that ever changes.
    """
    rules = [label.split("-")[0] for label in labels]
    n = len(labels)
    groups = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = tuple(sorted((rules[i], rules[j])))
            groups[key].append(similarities[i, j])
    return {key: float(np.mean(vals)) for key, vals in groups.items()}


def main():
    systems, labels = [], []
    for rule in RULES:
        for seed in SEEDS:
            run = load_run(seed, rule)
            systems.append(per_direction_trials(run))
            labels.append(f"{rule}-{seed}")

    similarities = analysis.run_dsa(systems)

    for pair, mean_dist in summarize_within_between(similarities, labels).items():
        print(f"{pair}: mean DSA distance = {mean_dist:.4f}")

    fig = analysis.plot_dsa_heatmap(similarities, labels,
                                     title="DSA: RFLO vs BPTT, all seeds")
    fig.savefig(RESULTS_DIR / "dsa_heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()
