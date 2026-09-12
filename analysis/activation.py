"""Activation / function-space channel: the primary, gauge-invariant lead.

Hidden-state trajectories H are PCA-reduced for visualization and compared with
DSA. DSA is legitimate here because activation trajectories are (closed-loop)
dynamical trajectories; it is not used on weights (see analysis.weights).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from DSA import DSA, pca_reduce

from .io import extract_per_direction_trajectories


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
