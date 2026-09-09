"""Grouped dispersion summaries and their permutation tests.

Pairwise distances from N systems are not independent samples, so group
differences (within-rule vs between-rule) are tested by permutation, never a
t-test. Two nulls live here: a free permutation over all systems, and a
block/paired permutation that respects the fact that BPTT-seedN and RFLO-seedN
share init (permute the rule label within each seed). Holm-Bonferroni corrects
across the family of cells.
"""

from collections import defaultdict
from itertools import combinations, product
from math import comb

import numpy as np


def summarize_within_between(similarities, labels, rules=None):
    """Mean pairwise DSA distance grouped by (rule, rule) pair."""
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
        # Even split guarantees both within-group keys plus one between-group key.
        assert len(summary.keys()) == 3
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


def calculate_p_value_of_dsa_distance(
    similarities, labels, rule_kinds=("BPTT", "RFLO"), n_mc_perms=10000, seed=42
):
    """Compute one-sided permutation p-value for within vs between group DSA distance.
    Uses exact enumeration if total combinations <= 10,000, otherwise Monte Carlo permutation sampling.
    """
    assert len(rule_kinds) == 2
    assert len(similarities) % 2 == 0
    n_systems = len(similarities)
    num_per_group = n_systems // 2

    true_summary = summarize_within_between(similarities, labels)
    test_stat = np.mean(
        [true_summary[(k1, k2)] for (k1, k2) in true_summary.keys() if k1 != k2]
    ) - np.mean(
        [true_summary[(k1, k2)] for (k1, k2) in true_summary.keys() if k1 == k2]
    )

    total_combs = comb(n_systems, num_per_group)

    if total_combs <= 10000:
        enumerated_metrics = calculate_stats_foreach_grouping(
            similarities, labels, rule_kinds
        )
        count = len([x for x in enumerated_metrics.values() if x >= test_stat])
        return (count + 1) / (len(enumerated_metrics) + 1)
    else:
        # Fast Monte Carlo permutation sampling
        rng = np.random.default_rng(seed)
        count = 0
        all_indices = np.arange(n_systems)
        sorted_rule_pair = tuple(sorted(rule_kinds))

        for _ in range(n_mc_perms):
            perm = rng.permutation(all_indices)
            group_a = set(perm[:num_per_group])
            rules = [
                rule_kinds[0] if i in group_a else rule_kinds[1]
                for i in range(n_systems)
            ]
            summary = summarize_within_between(similarities, labels, rules=rules)
            within_group = (
                summary.get((rule_kinds[0], rule_kinds[0]), 0)
                + summary.get((rule_kinds[1], rule_kinds[1]), 0)
            ) / 2.0
            between_group = summary.get(sorted_rule_pair, 0)
            perm_stat = between_group - within_group
            if perm_stat >= test_stat:
                count += 1
        return (count + 1) / (n_mc_perms + 1)


def _seed_to_pair(labels):
    """Map "RULE-SEED" labels to {seed: {rule: row_index}}."""
    pairs = defaultdict(dict)
    for idx, label in enumerate(labels):
        rule, seed = label.split("-")
        pairs[seed][rule] = idx
    return pairs


def _within_mean(M, idxs):
    pairs = list(combinations(idxs, 2))
    total = 0
    for p in pairs:
        total += M[p]
    return float(total / len(pairs))


def _paired_stat(pattern, seeds, pairs, M, s, rule_kinds):
    a_idxs, b_idxs = [], []
    for seed, bit in zip(seeds, pattern):
        pair = pairs[seed]
        if bit == 0:
            a_idxs.append(pair[rule_kinds[0]])
            b_idxs.append(pair[rule_kinds[1]])
        else:
            a_idxs.append(pair[rule_kinds[1]])
            b_idxs.append(pair[rule_kinds[0]])
    return s * (_within_mean(M, a_idxs) - _within_mean(M, b_idxs))


def paired_permutation_test(
    pairwise, labels, metric_is_similarity=False, rule_kinds=("BPTT", "RFLO")
):
    """One-sided paired permutation test for within-rule dispersion asymmetry.

    Permutes rule labels within each seed across all 2**n_seeds sign-flips.
    Positive statistic indicates RFLO clusters tighter than BPTT:
        s * (within_BPTT - within_RFLO), where s = -1 for similarity, +1 for distance.
    """
    s = -1 if metric_is_similarity else 1
    M = (np.asarray(pairwise) + np.asarray(pairwise).T) / 2
    pairs = _seed_to_pair(labels)
    seeds = sorted(pairs)

    observed = _paired_stat((0,) * len(seeds), seeds, pairs, M, s, rule_kinds)
    null = np.array(
        [
            _paired_stat(p, seeds, pairs, M, s, rule_kinds)
            for p in product((0, 1), repeat=len(seeds))
        ]
    )
    p_value = (np.sum(null >= observed) + 1) / (len(null) + 1)

    return {
        "observed": observed,
        "p_value": p_value,
        "null": null,
        "n_permutations": len(null),
    }


def holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni step-down correction, returning (reject, p_adjusted)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    # Scale by remaining tests; cummax enforces step-down monotonicity
    adjusted_sorted = np.clip(
        np.maximum.accumulate(ranked * (m - np.arange(m))), None, 1.0
    )
    p_adjusted = np.empty(m)
    p_adjusted[order] = adjusted_sorted
    return p_adjusted <= alpha, p_adjusted
