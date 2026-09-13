"""Artifact loading and the raw-H reshaping every channel starts from.

Deliberately decoupled from motornet: these functions read the self-describing
`.pt` artifacts that run_experiment.py writes and hand back plain dicts/arrays,
so analysis never needs the training environment on the path.
"""

from pathlib import Path

import numpy as np
import torch

from ._common import _to_numpy


def loss_windows(losses, window=2000):
    """Non-overlapping window means of a training-loss curve."""
    L = np.asarray(losses, dtype=float)
    return np.array(
        [L[i : i + window].mean() for i in range(0, len(L) - window + 1, window)]
    )


def converged(losses, window=2000, tol=2e-3, loss_gate=0.10):
    """Checks to see if a run converges. Loss needs to plateau and also be below 0.1."""
    w = loss_windows(losses, window)
    return bool(len(w) >= 2 and (w[-2] - w[-1]) < tol and w[-1] < loss_gate)


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
    """Reshape one saved run's H, shaped (T, n_targets, n_rec), into a list of
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
