"""Behavioral (reach-outcome) metrics

Everything here is expressed in, or converted to, a fraction of each effector's
reach distance. The two effectors live in different coordinate systems (arm in
metres, point mass in a dimensionless box), so any number compared across them
must be reach-relative, and not a shared absolute value.
"""

import numpy as np

from ._common import _to_numpy


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
    point mass and arm live on the same scale."""
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
