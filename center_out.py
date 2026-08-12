from ctd.task_modeling.task_env.random_target import RandomTarget
import matplotlib.pyplot as plt
import numpy as np
import torch
import motornet as mn  # noqa: E402

class CenterOut(RandomTarget):
    def __init__(self, effector, n_targets=8, target_distance=0.10,
                 center_sho_deg=67.5, center_elb_deg=77.5, *args, **kwargs):
        super().__init__(
            effector=effector,
            max_ep_duration=1.0, 
            bump_mag_low=0, bump_mag_high=0, # no perturbation
            *args, **kwargs
        )
        self.n_targets = n_targets
        self.target_distance = target_distance
        self.center_sho_ang = np.deg2rad(center_sho_deg)
        self.center_elb_ang = np.deg2rad(center_elb_deg)
        self.dataset_name = "CenterOut"

    def _center_angs(self):
        return torch.tensor(np.array([self.center_sho_ang, self.center_elb_ang, 0, 0]))

    def generate_trial_info(self):
        """Overrides RandomTarget.generate_trial_info: fixed center start,
        target drawn from n_targets evenly-spaced ring positions instead
        of a fully random joint-space target."""
        angs = self._center_angs()
        center_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0] # converts joint angles into actual positions in the plane
        direction_idx = np.random.randint(self.n_targets) # pick one of n_target indices
        theta = 2 * np.pi * direction_idx / self.n_targets # convert the pick into an angle
        offset = torch.tensor(
            [self.target_distance * np.cos(theta), self.target_distance * np.sin(theta)],
            dtype=torch.float32, device=self.device,
        ) # x and y coordinates of the target
        target_pos = center_xy + offset # centre offset on centre_xy
        return dict(ics_joint=angs, ics_xy=center_xy, goal=target_pos,
                    direction_idx=direction_idx)

    def visualise(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        angs = torch.tensor(np.array([self.center_sho_ang, self.center_elb_ang, 0, 0]))
        center_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0].detach().cpu().numpy().flatten()

        L1, L2 = self.effector.skeleton.L1, self.effector.skeleton.L2
        sho, elb = self.center_sho_ang, self.center_elb_ang
        shoulder_xy = np.array([0.0, 0.0])
        elbow_xy = np.array([L1 * np.cos(sho), L1 * np.sin(sho)])
        hand_xy = elbow_xy + np.array([L2 * np.cos(sho + elb), L2 * np.sin(sho + elb)])

        ax.plot([shoulder_xy[0], elbow_xy[0]], [shoulder_xy[1], elbow_xy[1]],
                '-', color='dimgray', linewidth=6, solid_capstyle='round', zorder=2)
        ax.plot([elbow_xy[0], hand_xy[0]], [elbow_xy[1], hand_xy[1]],
                '-', color='darkgray', linewidth=6, solid_capstyle='round', zorder=2)
        ax.scatter(*shoulder_xy, color='black', s=70, zorder=4, label='shoulder')
        ax.scatter(*elbow_xy, color='gray', s=40, zorder=4)
        ax.scatter(*hand_xy, color='tab:blue', s=70, zorder=5, label='start (fingertip)')
 
        # ring of possible targets
        thetas = 2 * np.pi * np.arange(self.n_targets) / self.n_targets
        target_x = hand_xy[0] + self.target_distance * np.cos(thetas)
        target_y = hand_xy[1] + self.target_distance * np.sin(thetas)
        ax.scatter(target_x, target_y, marker='x', color='tab:red', s=60, zorder=5, label='targets')
        for i, (tx, ty) in enumerate(zip(target_x, target_y)):
            ax.annotate(str(i), (tx, ty), textcoords="offset points", xytext=(5, 5), fontsize=8)
 
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'CenterOut initial state ({self.n_targets} targets, r={self.target_distance} m)')
        ax.legend(loc='best', fontsize=8)
        return fig, ax

from ctd.task_modeling.task_env.random_target import RandomTarget
import matplotlib.pyplot as plt
import numpy as np
import torch
import motornet as mn  # noqa: E402

class CenterOutV2(RandomTarget):
    def __init__(self, effector, n_targets=8, target_distance=0.10,
                 center_sho_deg=67.5, center_elb_deg=77.5, *args, **kwargs):
        self.n_targets = n_targets
        self.target_distance = target_distance
        self.center_sho_ang = np.deg2rad(center_sho_deg)
        self.center_elb_ang = np.deg2rad(center_elb_deg)
        self.dataset_name = "CenterOut"
        super().__init__(
            effector=effector, 
            max_ep_duration=1.0, 
            bump_mag_low=0, bump_mag_high=0, # no perturbation
            *args, **kwargs
        )

    def _center_angs(self):
        return torch.tensor(np.array([self.center_sho_ang, self.center_elb_ang, 0, 0]))

    def generate_trial_info(self):
        """Overrides RandomTarget.generate_trial_info: fixed center start,
        target drawn from n_targets evenly-spaced ring positions instead
        of a fully random joint-space target."""
        angs = self._center_angs()
        center_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0] # converts joint angles into actual positions in the plane
        direction_idx = np.random.randint(self.n_targets) # pick one of n_target indices
        theta = 2 * np.pi * direction_idx / self.n_targets # convert the pick into an angle
        offset = torch.tensor(
            [self.target_distance * np.cos(theta), self.target_distance * np.sin(theta)],
            dtype=torch.float32, device=self.device,
        ) # x and y coordinates of the target
        target_pos = center_xy + offset # centre offset on centre_xy
        return dict(ics_joint=angs, ics_xy=center_xy, goal=target_pos,
                    direction_idx=direction_idx)

    def reset(self, batch_size=1, options=None, seed=None):
        if options is not None and "batch_size" in options:
            batch_size = options["batch_size"]

        options = dict(options) if options is not None else {}
        if "ic_state" not in options or "target_state" not in options:
            ics_list, goal_list = [], []
            for _ in range(batch_size):
                info = self.generate_trial_info()
                ics_list.append(info["ics_joint"])
                goal_list.append(info["goal"])
            options.setdefault("ic_state", torch.stack(ics_list, dim=0).float())
            options.setdefault("target_state", torch.cat(goal_list, dim=0))

        return super().reset(batch_size=batch_size, options=options, seed=seed)

    def get_obs(self, action=None, deterministic: bool = False):
        """Overrides RandomTarget.get_obs(). The parent's obs is vision +
        proprioception only -- no target -- because the toolkit's own
        pipeline delivers the target separately (via generate_dataset()'s
        'inputs' tensor, concatenated externally). RFLOTrainer/BPTTTrainer
        use x_t = obs directly with no such external concatenation, so
        the target needs to be folded into obs itself instead."""
        obs = super().get_obs(action=action, deterministic=True)
        goal = self.goal if self.differentiable else self.detach(self.goal)
        obs = torch.cat([obs, goal], dim=-1)
        if not deterministic:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs


    def visualise(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        angs = torch.tensor(np.array([self.center_sho_ang, self.center_elb_ang, 0, 0]))
        center_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0].detach().cpu().numpy().flatten()

        L1, L2 = self.effector.skeleton.L1, self.effector.skeleton.L2
        sho, elb = self.center_sho_ang, self.center_elb_ang
        shoulder_xy = np.array([0.0, 0.0])
        elbow_xy = np.array([L1 * np.cos(sho), L1 * np.sin(sho)])
        hand_xy = elbow_xy + np.array([L2 * np.cos(sho + elb), L2 * np.sin(sho + elb)])

        ax.plot([shoulder_xy[0], elbow_xy[0]], [shoulder_xy[1], elbow_xy[1]],
                '-', color='dimgray', linewidth=6, solid_capstyle='round', zorder=2)
        ax.plot([elbow_xy[0], hand_xy[0]], [elbow_xy[1], hand_xy[1]],
                '-', color='darkgray', linewidth=6, solid_capstyle='round', zorder=2)
        ax.scatter(*shoulder_xy, color='black', s=70, zorder=4, label='shoulder')
        ax.scatter(*elbow_xy, color='gray', s=40, zorder=4)
        ax.scatter(*hand_xy, color='tab:blue', s=70, zorder=5, label='start (fingertip)')
 
        # ring of possible targets
        thetas = 2 * np.pi * np.arange(self.n_targets) / self.n_targets
        target_x = hand_xy[0] + self.target_distance * np.cos(thetas)
        target_y = hand_xy[1] + self.target_distance * np.sin(thetas)
        ax.scatter(target_x, target_y, marker='x', color='tab:red', s=60, zorder=5, label='targets')
        for i, (tx, ty) in enumerate(zip(target_x, target_y)):
            ax.annotate(str(i), (tx, ty), textcoords="offset points", xytext=(5, 5), fontsize=8)
 
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'CenterOut initial state ({self.n_targets} targets, r={self.target_distance} m)')
        ax.legend(loc='best', fontsize=8)
        return fig, ax