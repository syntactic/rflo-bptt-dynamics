from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from DSA import DSA, coarse_grain, pca_reduce


def _to_numpy(x):
    """Convert a torch tensor (on any device) or array-like to a numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_experiment_run(results_dir, effector, seed, rule):
    """Load and validate a single experiment artifact dictionary."""
    results_dir = Path(results_dir)
    file_path = results_dir / f"{effector}_seed{seed}_{rule}.pt"
    if not file_path.exists():
        raise FileNotFoundError(f"Experiment artifact not found at: {file_path}")

    data = torch.load(file_path, map_location="cpu")
    required_keys = (
        "weight_history",
        "losses",
        "metrics",
        "H",
        "direction_idx",
        "seed",
        "effector",
    )
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(f"Artifact {file_path} is missing expected keys: {missing_keys}")
    return data


def load_all_runs(results_dir, effector, seeds, rules=("BPTT", "RFLO")):
    """Load all experiment runs for a given effector, seed set, and rules."""
    runs = {}
    for rule in rules:
        for seed in seeds:
            runs[(rule, seed)] = load_experiment_run(results_dir, effector, seed, rule)
    return runs


def extract_per_direction_trajectories(run):
    """Reshape one saved run's H -- (T, n_targets, n_rec) -- into a list of
    (T, n_rec) per-direction trajectories for DSA.
    """
    H = run["H"]
    direction_idx = _to_numpy(run["direction_idx"]).squeeze()
    n_targets = H.shape[1]

    assert np.array_equal(direction_idx, np.arange(n_targets)), (
        f"H batch dim is not ordered 0..{n_targets - 1}: got {direction_idx}"
    )

    if torch.is_tensor(H):
        H_perm = H.permute(1, 0, 2).detach().cpu().numpy()
    else:
        H_perm = np.transpose(H, (1, 0, 2))
    return [H_perm[i] for i in range(n_targets)]


def process_activation_trajectories(runs, n_components=0.95):
    """Concatenate and PCA-reduce recurrent hidden state activations across runs and targets."""
    all_trajs = []
    keys = list(runs.keys())

    for key in keys:
        trials = extract_per_direction_trajectories(runs[key])  # list of (T, n_rec)
        stacked = np.concatenate(trials, axis=0)  # (T * n_targets, n_rec)
        all_trajs.append(stacked)

    combined_data = np.concatenate(all_trajs, axis=0)
    _, pca = pca_reduce(
        combined_data, n_components=n_components, return_pca=True, verbose=False
    )

    pc_trajectories = {}
    for key in keys:
        trials = extract_per_direction_trajectories(runs[key])
        pc_trajectories[key] = [pca.transform(traj) for traj in trials]

    return pc_trajectories, pca


def process_weight_trajectories(runs, bin_size=10, n_components=0.95):
    """Center, coarse-grain, and PCA-reduce recurrent weight trajectories across runs."""
    smoothed_trajectories = {}
    keys = list(runs.keys())

    for key in keys:
        raw_history = runs[key]["weight_history"]
        if torch.is_tensor(raw_history[0]):
            W_traj = torch.stack(raw_history).detach().cpu().numpy()
        else:
            W_traj = np.asarray(raw_history)
        centered = W_traj - W_traj[0:1, :]
        smoothed_trajectories[key] = coarse_grain(centered, bin_size=bin_size)

    all_data = np.concatenate([smoothed_trajectories[k] for k in keys], axis=0)
    _, pca = pca_reduce(
        all_data, n_components=n_components, return_pca=True, verbose=False
    )

    pc_trajectories = {}
    for key in keys:
        pc_trajectories[key] = pca.transform(smoothed_trajectories[key])

    return pc_trajectories, pca


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

    from math import comb

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


def run_dsa(
    latents,
    n_delays=10,
    rank=20,
    delay_interval=1,
    iters=500,
    lr=1e-2,
    score_method="angular",
    device="cpu",
):
    # Guard against short trajectory lengths (e.g. fast integration tests) where T - n_delays < rank
    sample_t = None
    if isinstance(latents, list) and len(latents) > 0:
        first = latents[0]
        if isinstance(first, list) and len(first) > 0:
            sample_t = len(first[0])
        elif hasattr(first, "shape"):
            sample_t = first.shape[0]
    elif hasattr(latents, "shape"):
        sample_t = latents.shape[0]

    if sample_t is not None:
        max_delays = max(1, sample_t // 2)
        n_delays = min(n_delays, max_delays)
        t_hankel = sample_t - (n_delays - 1) * delay_interval
        rank = min(rank, max(1, t_hankel - 2))

    dsa = DSA(
        latents,
        n_delays=n_delays,
        rank=rank,
        delay_interval=delay_interval,
        iters=iters,
        lr=lr,
        score_method=score_method,
        device=device,
    )
    return np.asarray(dsa.fit_score())


def plot_dsa_heatmap(similarities, labels, title="DSA: RFLO vs BPTT"):
    num_labels = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, num_labels), max(5, num_labels - 2)))
    sns.heatmap(
        similarities,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        ax=ax,
    )
    ax.collections[0].colorbar.ax.set_ylabel("DSA distance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def perp_dist(traj, a, b):
    """Perpendicular distance of each point in `traj` (T,2) from the straight
    line through points `a` and `b` (center and target)."""
    ab = b - a
    L = np.linalg.norm(ab)
    if L == 0:
        return np.zeros(len(traj))
    ap = traj - a
    cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
    return np.abs(cross) / L


# Cartesian reach distance per effector, in each effector's own position units.
# Source of truth is run_experiment.py, which sets the CenterOutReach `n` as
# `0.1 if "Arm" in effector else 0.5` (arm in metres ~= 10 cm reach; point mass in a
# dimensionless +/-1 box). These units differ, so any cross-effector behavioral number
# (terminal error, success bar) must be expressed as a FRACTION of this distance, never
# as a shared absolute value.
def reach_distance(effector):
    """Reach distance (target radius from center) for an effector, matching the value
    run_experiment.py trained with. Used to convert absolute terminal error into a
    reach-relative fraction so point mass (0.5) and arm (0.1) are comparable."""
    return 0.1 if "Arm" in effector else 0.5


def terminal_error(FT, targets):
    """Per-target Euclidean distance between the final fingertip position and the target,
    in the effector's own units (raw, unscaled).

    FT is batch-first (n_targets, T, 2); targets is (n_targets, 2). Returns a
    (n_targets,) array. Mirrors the endpoint math in diagnose_endpoints.py; divide by
    reach_distance(effector) for the reach-relative metric that the behavior-matching
    gate compares across rules.
    """
    FT = _to_numpy(FT)
    targets = _to_numpy(targets)
    return np.linalg.norm(FT[:, -1, :] - targets, axis=-1)


def reach_relative_terminal_error(FT, targets, effector):
    """terminal_error expressed as a fraction of the effector's reach distance, so the
    point mass and arm live on the same scale. This is the primary behavioral metric for
    comparing whether two learning rules reach matched behavior (a continuous distribution,
    not a binary success rate)."""
    return terminal_error(FT, targets) / reach_distance(effector)


def reach_metrics(FT, targets, center=None, success_radius=0.05):
    """Per-target success (did the final fingertip land within `success_radius`
    of the target?) and mean straight-line path deviation. targets is an array of
    pairs and center is a single pair.

    Returns (success_rate in [0,1], mean_path_deviation).
    """
    targets = _to_numpy(targets)
    FT = _to_numpy(FT)
    if center is not None:
        starts = np.broadcast_to(_to_numpy(center), targets.shape)
    else:
        starts = FT[:, 0, :]
    dev = []
    final_pos = FT[:, -1, :]
    succ = np.linalg.norm(final_pos - targets, axis=-1) < success_radius
    for i in range(FT.shape[0]):
        dev.append(perp_dist(FT[i], starts[i], targets[i]).mean())
    return float(np.mean(succ)), float(np.mean(dev))


def per_direction_metrics(FT, targets, direction_idx, success_radius=0.05):
    """Per direction success, assumes that we're doing center out because we have direction indices"""

    targets = _to_numpy(targets)
    FT = _to_numpy(FT)
    direction_idx = _to_numpy(direction_idx.squeeze())
    metrics = {}
    for i in np.unique(direction_idx):
        FT_subset = FT[direction_idx == i]
        targets_subset = targets[direction_idx == i]
        metrics[int(i)] = reach_metrics(
            FT_subset, targets_subset, success_radius=success_radius
        )

    return metrics
