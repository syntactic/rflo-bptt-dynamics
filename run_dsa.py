"""Activation-space DSA between BPTT and RFLO across seeds saved by run_experiment.py."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import analysis
import argparse

RESULTS_DIR = Path("results")
SEEDS = range(5)
RULES = ("BPTT", "RFLO")


def load_run(effector, seed, rule):
    return torch.load(RESULTS_DIR / f"{effector}_seed{seed}_{rule}.pt")


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


def summarize_within_between(similarities, labels, rules=None):
    """Mean pairwise DSA distance grouped by (rule, rule) pair -- the numeric
    version of "do same-rule runs cluster together" that's easy to eyeball on
    a small heatmap but hard to judge reliably as it grows (10x10 already
    strains it). Groups on the rule prefix of each label (e.g. "BPTT-3" ->
    "BPTT"), so this generalizes past exactly two rules if that ever changes.
    """
    n = len(labels)
    groups = defaultdict(list)
    if rules is None:
        rules = [label.split("-")[0] for label in labels]
    assert n == len(rules)
    assert len(similarities) == len(rules)
    for i in range(n):
        for j in range(i + 1, n):
            key = tuple(sorted((rules[i], rules[j])))
            groups[key].append(similarities[i, j])
    return {key: float(np.mean(vals)) for key, vals in groups.items()}


def calculate_stats_foreach_grouping(similarities, labels, rule_kinds):
    assert len(rule_kinds) == 2
    assert len(similarities) % 2 == 0
    num_per_group = int(len(similarities) / 2)
    group_a_assignments = combinations(range(len(similarities)), num_per_group)
    metrics_per_assignment = {}
    for assignment in group_a_assignments:
        rules = []
        for i in range(len(similarities)):
            if i in assignment:
                rules.append(rule_kinds[0])
            else:
                rules.append(rule_kinds[1])
        summary = summarize_within_between(similarities, labels, rules=rules)
        assert (
            len(summary.keys()) == 3
        )  # two within group measures and one between group measures
        within_group = 0
        between_group = 0
        for (r1, r2), mean_similarity in summary.items():
            if r1 == r2:
                within_group += mean_similarity
            else:
                between_group += mean_similarity
        within_group /= len(rule_kinds)
        metrics_per_assignment[assignment] = between_group - within_group
    return metrics_per_assignment


def calculate_p_value_of_dsa_distance(similarities, labels, rule_kinds):
    enumerated_metrics = calculate_stats_foreach_grouping(
        similarities, labels, rule_kinds
    )
    true_summary = summarize_within_between(similarities, labels)
    test_stat = np.mean(
        [true_summary[(k1, k2)] for (k1, k2) in true_summary.keys() if k1 != k2]
    ) - np.mean(
        [true_summary[(k1, k2)] for (k1, k2) in true_summary.keys() if k1 == k2]
    )
    return (len([x for x in enumerated_metrics.values() if x >= test_stat]) + 1) / (
        len(enumerated_metrics) + 1
    )


def main():
    parser = argparse.ArgumentParser(
        prog="DSA Runner",
        description="Analyze dynamics between systems",
    )
    parser.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"])
    args = parser.parse_args()

    systems, labels = [], []
    for rule in RULES:
        for seed in SEEDS:
            run = load_run(args.effector, seed, rule)
            systems.append(per_direction_trials(run))
            labels.append(f"{rule}-{seed}")

    similarities = analysis.run_dsa(systems)

    for pair, mean_dist in summarize_within_between(similarities, labels).items():
        print(f"{pair}: mean DSA distance = {mean_dist:.4f}")

    print(f"p-value:", calculate_p_value_of_dsa_distance(similarities, labels, RULES))

    fig = analysis.plot_dsa_heatmap(
        similarities, labels, title="DSA: RFLO vs BPTT, all seeds"
    )
    fig.savefig(RESULTS_DIR / f"{args.effector}_dsa_heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()
