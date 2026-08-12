import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_workspace_and_targets(
    axis,
    effector,
    start: torch.Tensor | np.ndarray,
    goals: torch.Tensor | np.ndarray,
    grid_resolution: int = 50,
) -> None:
  """Plots an effector's reachable cartesian workspace as a point cloud, overlaid with a starting
  position and a set of target positions — a sanity check that targets fall inside the reachable region.

  Args:
    axis: A `matplotlib` axis handle.
    effector: :class:`motornet.effector.Effector` object whose workspace is being visualized. Must have
      `dof == 2` (this function only supports 2-DOF effectors).
    start: `Tensor` or `numpy.ndarray` of shape `(batch_size, 2)`, the starting cartesian position(s).
    goals: `Tensor` or `numpy.ndarray` of shape `(batch_size, 2)`, the target cartesian position(s).
    grid_resolution: `Integer`, the number of samples per joint dimension used to build the workspace grid.
  """
  assert effector.dof == 2, "plot_workspace_and_targets only supports 2-DOF effectors."

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

  start_xy = (start.detach().cpu().numpy() if torch.is_tensor(start) else np.array(start)).reshape(-1, 2)
  goals_xy = (goals.detach().cpu().numpy() if torch.is_tensor(goals) else np.array(goals)).reshape(-1, 2)

  axis.scatter(workspace_xy[:, 0], workspace_xy[:, 1], s=1, c='lightgray', label='reachable workspace')
  axis.scatter(start_xy[:, 0], start_xy[:, 1], c='black', marker='o', s=60, label='start')
  axis.scatter(goals_xy[:, 0], goals_xy[:, 1], c='red', marker='x', s=60, label='targets')
  axis.set_xlabel('cartesian x')
  axis.set_ylabel('cartesian y')
  axis.set_aspect('equal', adjustable='box')
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
        The `matplotlib.figure.Figure`, with three panels -- training loss,
        mean success rate, and mean path deviation, each over training step --
        one line per label in `runs`. Success rate/deviation are averaged
        across directions per checkpoint; per-direction detail is not shown here.
    """
    fig, (loss_ax, succ_ax, dev_ax) = plt.subplots(1, 3, figsize=(15, 4))

    for label, (losses, metrics) in runs.items():
        loss_ax.plot(losses, label=label)

        steps = [step for step, _ in metrics]
        mean_success = [np.mean([v[0] for v in per_dir.values()]) for _, per_dir in metrics]
        mean_deviation = [np.mean([v[1] for v in per_dir.values()]) for _, per_dir in metrics]

        succ_ax.plot(steps, mean_success, marker='o', label=label)
        dev_ax.plot(steps, mean_deviation, marker='o', label=label)

    loss_ax.set_xlabel('training step')
    loss_ax.set_ylabel('loss')
    loss_ax.set_title('training loss')
    loss_ax.legend()

    succ_ax.set_xlabel('training step')
    succ_ax.set_ylabel('mean success rate (across directions)')
    succ_ax.set_title('behavioral success rate')
    succ_ax.set_ylim(-0.05, 1.05)
    succ_ax.legend()

    dev_ax.set_xlabel('training step')
    dev_ax.set_ylabel('mean path deviation (across directions)')
    dev_ax.set_title('behavioral path deviation')
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
        (succ_ax, 'success rate by direction', 'success rate'),
        (dev_ax, 'path deviation by direction', 'mean path deviation'),
    ]:
        a.set_xticks(x)
        a.set_xticklabels(directions)
        a.set_xlabel('direction index')
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.legend()

    fig.tight_layout()
    return fig

def plot_reach_trajectories(FT, goal, direction_idx):
    plt.figure(figsize=(6, 6))
    dynamic_cmap = plt.colormaps['turbo'].resampled(len(np.unique(direction_idx)))

    for i in range(len(FT)):
        plt.plot(FT[i, :, 0], FT[i, :, 1], color=dynamic_cmap(direction_idx[i]), lw=1.5)  # path
        plt.plot(FT[i, 0, 0],  FT[i, 0, 1],  'o', color=dynamic_cmap(direction_idx[i]), ms=6)   # start
        plt.plot(FT[i, -1, 0], FT[i, -1, 1], 's', color=dynamic_cmap(direction_idx[i]), ms=6)   # end
        plt.plot(goal[i, 0], goal[i, 1], '*', color=dynamic_cmap(direction_idx[i]), ms=16, mec='k')  # target
    plt.show()
