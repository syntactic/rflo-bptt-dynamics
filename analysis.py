import numpy as np
import torch
import seaborn as sns
from DSA import DSA
import matplotlib.pyplot as plt

def _to_numpy(x):
    """Convert a torch tensor (on any device) or array-like to a numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def run_dsa(latents, n_delays=10, rank=20, delay_interval=1,
            iters=500, lr=1e-2, score_method="angular", device="cpu"):
    dsa = DSA(latents, n_delays=n_delays, rank=rank, delay_interval=delay_interval,
              iters=iters, lr=lr, score_method=score_method, device=device)
    return np.asarray(dsa.fit_score())


def plot_dsa_heatmap(similarities, labels, title="DSA: RFLO vs BPTT"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(similarities, xticklabels=labels, yticklabels=labels,
                annot=True, fmt=".3f", ax=ax)
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
        metrics[int(i)] = reach_metrics(FT_subset, targets_subset, success_radius=success_radius)

    return metrics
    
