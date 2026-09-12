import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_workspace_and_targets(
    axis,
    effector,
    start: torch.Tensor | np.ndarray,
    goals: torch.Tensor | np.ndarray,
    grid_resolution: int = 50,
) -> None:
    """Plots an effector's reachable cartesian workspace as a point cloud, overlaid with a starting
    position and a set of target positions. A sanity check that targets fall inside the reachable region.

    Args:
      axis: A `matplotlib` axis handle.
      effector: :class:`motornet.effector.Effector` object whose workspace is being visualized. Must have
        `dof == 2` (this function only supports 2-DOF effectors).
      start: `Tensor` or `numpy.ndarray` of shape `(batch_size, 2)`, the starting cartesian position(s).
      goals: `Tensor` or `numpy.ndarray` of shape `(batch_size, 2)`, the target cartesian position(s).
      grid_resolution: `Integer`, the number of samples per joint dimension used to build the workspace grid.
    """
    assert effector.dof == 2, (
        "plot_workspace_and_targets only supports 2-DOF effectors."
    )

    lb = effector.pos_lower_bound.detach().cpu().numpy()
    ub = effector.pos_upper_bound.detach().cpu().numpy()
    q0, q1 = np.meshgrid(
        np.linspace(lb[0], ub[0], grid_resolution),
        np.linspace(lb[1], ub[1], grid_resolution),
    )
    positions = np.stack([q0.ravel(), q1.ravel()], axis=1)
    velocities = np.zeros_like(positions)
    joint_state = torch.tensor(
        np.concatenate([positions, velocities], axis=1), dtype=torch.float32
    ).to(effector.device)

    cartesian = effector.joint2cartesian(joint_state)
    workspace_xy = cartesian.chunk(2, dim=-1)[0].detach().cpu().numpy()

    start_xy = (
        start.detach().cpu().numpy() if torch.is_tensor(start) else np.array(start)
    ).reshape(-1, 2)
    goals_xy = (
        goals.detach().cpu().numpy() if torch.is_tensor(goals) else np.array(goals)
    ).reshape(-1, 2)

    axis.scatter(
        workspace_xy[:, 0],
        workspace_xy[:, 1],
        s=1,
        c="lightgray",
        label="reachable workspace",
    )
    axis.scatter(
        start_xy[:, 0], start_xy[:, 1], c="black", marker="o", s=60, label="start"
    )
    axis.scatter(
        goals_xy[:, 0], goals_xy[:, 1], c="red", marker="x", s=60, label="targets"
    )
    axis.set_xlabel("cartesian x")
    axis.set_ylabel("cartesian y")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()


def plot_training_curves(runs):
    """Compare loss and behavioral learning curves across trainers.

    Args:
        runs: `dict` mapping a label (e.g. "RFLO", "BPTT") to that trainer's
          `(losses, metrics)` tuple, exactly as returned by `BaseTrainer.train()`.
          `metrics` is the list of `(step, per_direction_dict)` pairs produced by
          `checkpoint_behavior()`, where `per_direction_dict` maps
          `direction_idx -> (success_rate, mean_deviation)`.

    Returns:
        The `matplotlib.figure.Figure`, with three panels (training loss,
        mean success rate, and mean path deviation, each over training step),
        one line per label in `runs`. Success rate/deviation are averaged
        across directions per checkpoint; per-direction detail is not shown here.
    """
    fig, (loss_ax, succ_ax, dev_ax) = plt.subplots(1, 3, figsize=(15, 4))

    for label, (losses, metrics) in runs.items():
        loss_ax.plot(losses, label=label)

        steps = [step for step, _ in metrics]
        mean_success = [
            np.mean([v[0] for v in per_dir.values()]) for _, per_dir in metrics
        ]
        mean_deviation = [
            np.mean([v[1] for v in per_dir.values()]) for _, per_dir in metrics
        ]

        succ_ax.plot(steps, mean_success, marker="o", label=label)
        dev_ax.plot(steps, mean_deviation, marker="o", label=label)

    loss_ax.set_xlabel("training step")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("training loss")
    loss_ax.legend()

    succ_ax.set_xlabel("training step")
    succ_ax.set_ylabel("mean success rate (across directions)")
    succ_ax.set_title("behavioral success rate")
    succ_ax.set_ylim(-0.05, 1.05)
    succ_ax.legend()

    dev_ax.set_xlabel("training step")
    dev_ax.set_ylabel("mean path deviation (across directions)")
    dev_ax.set_title("behavioral path deviation")
    dev_ax.legend()

    fig.tight_layout()
    return fig


def plot_per_direction_comparison(runs):
    """Grouped bar chart comparing per-direction behavioral metrics across trainers.

    Args:
        runs: `dict` mapping a label (e.g. "RFLO", "BPTT") to a per-direction
          metrics dict as returned by `per_direction_metrics()` /
          `BaseTrainer.checkpoint_behavior()`:
          `direction_idx -> (success_rate, mean_deviation)`.

    Returns:
        The `matplotlib.figure.Figure`, with two panels (success rate, path
        deviation), each showing one bar group per direction, one bar per
        label in `runs`.
    """
    labels = list(runs.keys())
    directions = sorted(set().union(*(d.keys() for d in runs.values())))
    n_labels = len(labels)
    width = 0.8 / n_labels
    x = np.arange(len(directions))

    fig, (succ_ax, dev_ax) = plt.subplots(1, 2, figsize=(12, 4))

    for j, label in enumerate(labels):
        per_dir = runs[label]
        success = [per_dir[d][0] for d in directions]
        deviation = [per_dir[d][1] for d in directions]
        offset = (j - (n_labels - 1) / 2) * width
        succ_ax.bar(x + offset, success, width, label=label)
        dev_ax.bar(x + offset, deviation, width, label=label)

    for a, title, ylabel in [
        (succ_ax, "success rate by direction", "success rate"),
        (dev_ax, "path deviation by direction", "mean path deviation"),
    ]:
        a.set_xticks(x)
        a.set_xticklabels(directions)
        a.set_xlabel("direction index")
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.legend()

    fig.tight_layout()
    return fig


def plot_arm_skeleton_and_targets(
    axis,
    effector,
    center_sho_deg: float = 67.5,
    center_elb_deg: float = 77.5,
    target_distance: float = 0.1,
    n_targets: int = 8,
) -> None:
    """Plots a 2-DOF arm's skeleton (shoulder, elbow, fingertip) and its center-out target ring."""
    sho, elb = np.deg2rad(center_sho_deg), np.deg2rad(center_elb_deg)
    L1 = getattr(effector.skeleton, "L1", 0.309)
    L2 = getattr(effector.skeleton, "L2", 0.26)

    shoulder_xy = np.array([0.0, 0.0])
    elbow_xy = np.array([L1 * np.cos(sho), L1 * np.sin(sho)])
    hand_xy = elbow_xy + np.array([L2 * np.cos(sho + elb), L2 * np.sin(sho + elb)])

    axis.plot(
        [shoulder_xy[0], elbow_xy[0]],
        [shoulder_xy[1], elbow_xy[1]],
        "-",
        color="dimgray",
        linewidth=5,
        solid_capstyle="round",
        zorder=2,
        label="upper arm",
    )
    axis.plot(
        [elbow_xy[0], hand_xy[0]],
        [elbow_xy[1], hand_xy[1]],
        "-",
        color="darkgray",
        linewidth=4,
        solid_capstyle="round",
        zorder=2,
        label="forearm",
    )
    axis.scatter(*shoulder_xy, color="black", s=60, zorder=4, label="shoulder")
    axis.scatter(*elbow_xy, color="gray", s=40, zorder=4)
    axis.scatter(*hand_xy, color="tab:blue", s=60, zorder=5, label="start (fingertip)")

    thetas = 2 * np.pi * np.arange(n_targets) / n_targets
    target_x = hand_xy[0] + target_distance * np.cos(thetas)
    target_y = hand_xy[1] + target_distance * np.sin(thetas)
    axis.scatter(
        target_x, target_y, marker="x", color="tab:red", s=50, zorder=5, label="targets"
    )
    for i, (tx, ty) in enumerate(zip(target_x, target_y)):
        axis.annotate(
            str(i), (tx, ty), textcoords="offset points", xytext=(4, 4), fontsize=8
        )

    axis.set_aspect("equal")
    axis.set_xlabel("cartesian x (m)")
    axis.set_ylabel("cartesian y (m)")
    axis.set_title(
        f"Center-Out Task Setup ({n_targets} targets, r={target_distance} m)"
    )
    axis.legend(loc="best", fontsize=8)


def plot_reach_trajectories(
    FT, goal, direction_idx, ax=None, title="Center-Out Reach Trajectories"
):
    """Plot reaching trajectories colored by target direction."""
    FT = FT.detach().cpu().numpy() if torch.is_tensor(FT) else np.asarray(FT)
    goal = goal.detach().cpu().numpy() if torch.is_tensor(goal) else np.asarray(goal)
    direction_idx = (
        direction_idx.detach().cpu().numpy().squeeze()
        if torch.is_tensor(direction_idx)
        else np.asarray(direction_idx).squeeze()
    )

    # Ensure FT is shaped (batch, time, 2)
    if (
        FT.ndim == 3
        and FT.shape[0] != len(direction_idx)
        and FT.shape[1] == len(direction_idx)
    ):
        FT = np.transpose(FT, (1, 0, 2))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    n_unique_dirs = len(np.unique(direction_idx))
    dynamic_cmap = plt.colormaps["turbo"].resampled(n_unique_dirs)

    for i in range(len(FT)):
        dir_val = (
            int(direction_idx[i]) if direction_idx.ndim > 0 else int(direction_idx)
        )
        color = dynamic_cmap(dir_val % n_unique_dirs)
        ax.plot(FT[i, :, 0], FT[i, :, 1], color=color, lw=1.5, alpha=0.8)
        ax.plot(FT[i, 0, 0], FT[i, 0, 1], "o", color=color, ms=4)
        ax.plot(FT[i, -1, 0], FT[i, -1, 1], "s", color=color, ms=4)
        ax.plot(goal[i, 0], goal[i, 1], "*", color=color, ms=10, mec="k", mew=0.5)

    ax.set_xlabel("cartesian x")
    ax.set_ylabel("cartesian y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_activation_pca(
    pc_trajectories, pca=None, title="Recurrent Hidden Dynamics (PCA)", ax=None
):
    """Plot PCA projection of hidden activations colored by reach direction."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # pc_trajectories is a dict: (rule, seed) -> list of per-direction (T, n_pcs) arrays
    # or a list of per-direction arrays for a single run
    if isinstance(pc_trajectories, dict):
        key = list(pc_trajectories.keys())[0]
        trajs = pc_trajectories[key]
    else:
        trajs = pc_trajectories

    dynamic_cmap = plt.colormaps["turbo"].resampled(len(trajs))
    for i, traj in enumerate(trajs):
        color = dynamic_cmap(i)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            lw=1.5,
            label=f"Dir {i}" if len(trajs) <= 8 else None,
        )
        ax.scatter(traj[0, 0], traj[0, 1], color="black", s=15, zorder=4)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="s", s=25, zorder=4)

    var1 = (
        f" ({pca.explained_variance_ratio_[0] * 100:.1f}%)" if pca is not None else ""
    )
    var2 = (
        f" ({pca.explained_variance_ratio_[1] * 100:.1f}%)" if pca is not None else ""
    )
    ax.set_xlabel(f"PC 1{var1}")
    ax.set_ylabel(f"PC 2{var2}")
    ax.set_title(title)
    if len(trajs) <= 8:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_weight_pca(
    pc_trajectories,
    pca=None,
    title="Weight Trajectories (W_rec): BPTT vs RFLO",
    ax=None,
):
    """Plot recurrent weight trajectories in PC space across seeds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    bptt_labeled, rflo_labeled = False, False
    for (rule, seed), traj in pc_trajectories.items():
        color = "tab:blue" if rule == "BPTT" else "tab:orange"
        label = None
        if rule == "BPTT" and not bptt_labeled:
            label = "BPTT"
            bptt_labeled = True
        elif rule == "RFLO" and not rflo_labeled:
            label = "RFLO"
            rflo_labeled = True

        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.8, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color="black", s=15, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="x", s=35, zorder=5)

    var1 = (
        f" ({pca.explained_variance_ratio_[0] * 100:.1f}%)" if pca is not None else ""
    )
    var2 = (
        f" ({pca.explained_variance_ratio_[1] * 100:.1f}%)" if pca is not None else ""
    )
    ax.set_xlabel(f"PC 1{var1}")
    ax.set_ylabel(f"PC 2{var2}")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
