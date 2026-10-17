"""Build the README and abstract figures from the exported data release in data/.

Every panel reads data/ through data_release, so the figures rebuild from a clone without
the run artifacts. Regenerate data/ with export_data.py when runs change.

README figure (figures/canalization_summary.png):
  A  both rules drive the arm to all eight targets, RFLO with larger errors.
  B  learning subspaces of RFLO seeds sharing one feedback matrix are far from
     orthogonal; the other conditions sit near the random-subspace null.
  C  RFLO's per-direction deficit is bimodal: each seed is worst toward direction 0
     or direction 4, roughly half the seeds each, against BPTT's flat profile.

Abstract figure (figures/abstract_summary.png) adds:
  D  training loss at 40k, COMMON17          E  training loss to 100k and 300k
  F  mean error at equal budgets             G  worst direction across nets x B's
  H  fixB per-direction error by budget      I  point-mass control on the arm's scale

Each panel is also saved alone as figures/panel_<name>.png.

Usage:
    python make_readme_figure.py [--data data] [--seed 1]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import data_release as dr

FIGURES = Path("figures")
ARM, PM = "RigidTendonArm26", "ReluPointMass24"

# One color per condition across every panel (checked for color-vision deficiency).
# The multiB draws are fixB-type conditions (a different shared B), so they share its hue.
COLOR = {
    "BPTT": "#1f6fb4",
    "projB0": "#c2185b",
    "fixB": "#d97000",
    "varyB": "#2a9d8f",
    "b101": "#d97000",
    "b102": "#d97000",
    "b103": "#d97000",
}
NAME = {
    "BPTT": "BPTT",
    "projB0": "projected BPTT",
    "fixB": "RFLO, shared B",
    "varyB": "RFLO, per-seed B",
    "b101": "RFLO, B 101",
    "b102": "RFLO, B 102",
    "b103": "RFLO, B 103",
}
INK, MUTED = "#222222", "#777777"
ERROR_YLIM = (0, 0.62)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
    }
)


class Data:
    """The tables the panels draw from, loaded once."""

    def __init__(self, data_dir):
        self.dir = data_dir
        self.runs = dr.load_runs(data_dir)
        self.curves = dr.load_loss_curves(data_dir)
        self.windows = dr.load_windows(data_dir).merge(
            self.runs[["run_id", "effector", "condition", "seed"]], on="run_id"
        )
        self.pairs = dr.load_pairwise_subspace(data_dir)

    def select(self, effector, condition, budget, seeds=None):
        return dr.select(self.runs, effector, condition, budget, seeds)

    def profiles(self, effector, condition, budget, seeds=None):
        sel = self.select(effector, condition, budget, seeds)
        sel = sel[sel.has_eval]
        return sel[dr.ERROR_COLUMNS].to_numpy(), list(sel.seed)


def direction_axis(ax, compact=False):
    ax.set_xticks(range(dr.N_DIRECTIONS))
    if compact:
        # Narrow panels: vertical labels, so eight of them fit.
        ax.set_xticklabels(
            [dr.direction_label(d) for d in range(dr.N_DIRECTIONS)],
            rotation=90,
            fontsize=6.5,
        )
    else:
        ax.set_xticklabels(
            [dr.direction_label(d) for d in range(dr.N_DIRECTIONS)],
            rotation=45,
            ha="right",
        )
    ax.set_xlabel("target direction (index, angle)")


def label_n(name, n):
    return f"{name} (n={n})"


# --- README panels -----------------------------------------------------------------------


def draw_reaches(ax, D, seed=1):
    """Fingertip paths for one seed under each rule, targets labeled by direction index."""
    for condition in ("BPTT", "fixB"):
        arr = dr.load_arrays(f"arm_{condition}_40k", D.dir)
        i = arr["run_id"].index(f"arm_{condition}_40k_s{seed}")
        FT, goal = arr["FT"][i], arr["goal"][i]
        for d in range(FT.shape[1]):
            ax.plot(
                FT[:, d, 0],
                FT[:, d, 1],
                color=COLOR[condition],
                lw=1.4,
                alpha=0.85,
                label=NAME[condition] if d == 0 else None,
            )
    start = FT[0, 0]
    ax.scatter(
        goal[:, 0],
        goal[:, 1],
        marker="*",
        s=90,
        facecolor="none",
        edgecolor="black",
        linewidths=0.8,
        zorder=5,
        label="targets",
    )
    ax.scatter(*start, color="black", s=25, zorder=6)
    for d, g in enumerate(goal):
        outward = (g - start) / np.linalg.norm(g - start)
        x, y = g + 0.022 * outward
        ax.text(x, y, str(d), ha="center", va="center", fontsize=8, color=INK)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("hand position x (m)")
    ax.set_ylabel("hand position y (m)")
    ax.set_title(f"Reaching behavior at 40k, seed {seed}", loc="left")
    ax.legend(loc="lower left", fontsize=7, frameon=False)


CORRIDOR_ITEMS = [
    # (label, effector, condition, steps, seeds)
    ("BPTT", ARM, "BPTT", (0, 40000), dr.COMMON17),
    ("projected BPTT", ARM, "projB0", (0, 40000), dr.COMMON17),
    ("RFLO, shared B", ARM, "fixB", (0, 40000), dr.COMMON17),
    ("RFLO, per-seed B", ARM, "varyB", (0, 40000), dr.COMMON17),
    # The multiB draws are on nets 1 3 5 8 only, so B0 on the same nets sits beside them.
    ("B₀", ARM, "fixB", (0, 40000), dr.GRID4),
    ("B 101", ARM, "b101", (0, 40000), dr.GRID4),
    ("B 102", ARM, "b102", (0, 40000), dr.GRID4),
    ("B 103", ARM, "b103", (0, 40000), dr.GRID4),
    ("BPTT", PM, "BPTT", (0, 70000), list(range(15))),
    ("RFLO, shared B", PM, "fixB", (0, 70000), list(range(15))),
]


def within_pair_angles(D, effector, condition, steps, seeds):
    w = D.windows
    ids = set(
        w[
            (w.effector == effector)
            & (w.condition == condition)
            & (w.step_start == steps[0])
            & (w.step_stop == steps[1])
            & w.seed.isin(seeds)
        ].window_id
    )
    p = D.pairs
    return p[p.window_i.isin(ids) & p.window_j.isin(ids)].angle_deg_k3.to_numpy()


def draw_corridor(ax, D, items=CORRIDOR_ITEMS[:4], with_controls=False):
    """Within-condition learning-subspace angle (k = 3): every seed pair plus the mean."""
    items = CORRIDOR_ITEMS if with_controls else items
    rng = np.random.default_rng(0)  # jitter only
    x = 0.0
    ticks = []
    for label, effector, condition, steps, seeds in items:
        if (effector == PM and not any(t[2] == PM for t in ticks)) or (
            seeds is dr.GRID4 and not any(t[3] is dr.GRID4 for t in ticks)
        ):
            x += 0.6  # gap before the multiB group and before the control
        angles = within_pair_angles(D, effector, condition, steps, seeds)
        color = COLOR[condition]
        hollow = effector == PM
        ax.scatter(
            x + rng.uniform(-0.18, 0.18, len(angles)),
            angles,
            s=6,
            color="none" if hollow else color,
            edgecolors=color,
            linewidths=0.5,
            alpha=0.45,
        )
        mean = angles.mean()
        ax.plot([x - 0.3, x + 0.3], [mean, mean], color=INK, lw=1.6)
        ax.text(x, 101, f"{mean:.1f}°", ha="center", fontsize=7.5, color=INK)
        ticks.append((x, label, effector, seeds))
        x += 1
    null = dr.random_null_deg()
    ax.axhline(null, ls="--", color="black", lw=1)
    # Placed over the shared-B columns, where no point reaches the null.
    ax.text(2.0, null + 1.0, f"random subspaces {null:.1f}°",
            ha="center", va="bottom", fontsize=7, color=INK)  # fmt: skip
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=7, rotation=40, ha="right")
    if with_controls:
        grid = [t[0] for t in ticks if t[1].startswith("B")]
        ax.text(
            np.mean(grid),
            30,
            "multiB, nets\n1 3 5 8",
            ha="center",
            fontsize=7,
            color=MUTED,
        )
        pm = [t[0] for t in ticks if t[2] == PM]
        ax.text(
            np.mean(pm),
            30,
            "point mass\n(control)",
            ha="center",
            fontsize=7,
            color=MUTED,
        )
    ax.set_ylim(25, 106)
    ax.set_ylabel("angle between seeds' learning subspaces (°)")
    # Not "sharing B aligns": projected BPTT is confined to span(B0) and stays near the null.
    ax.set_title("Shared B plus RFLO's credit aligns the updates", loc="left")


def pole_groups(profiles):
    """Split seeds by the pole of the 0-4 axis they are worst toward: (dir 4, dir 0).

    The deficit is bimodal, so a mean over seeds averages two anti-correlated shapes into
    a double bump no individual seed has. The worst direction gives the cited 11/6 split
    for fixB, as does PC1 fitted on the pooled fixB and varyB profiles; PC1 fitted on fixB
    alone moves borderline seed 21 and gives 10/7. Seeds worst elsewhere are left out.
    """
    worst = profiles.argmax(axis=1)
    return [worst == 4, worst == 0]


def draw_profiles(ax, D):
    """Per-direction error: the two RFLO subtypes against BPTT's flat profile (40k)."""
    fixB, _ = D.profiles(ARM, "fixB", 40000, dr.COMMON17)
    directions = np.arange(dr.N_DIRECTIONS)
    for mask, style, pole in zip(pole_groups(fixB), ("-o", "-s"), (4, 0)):
        group = fixB[mask]
        mean, sem = group.mean(0), group.std(0) / np.sqrt(len(group))
        ax.errorbar(
            directions,
            mean,
            yerr=sem,
            fmt=style,
            color=COLOR["fixB"],
            markerfacecolor=COLOR["fixB"] if style == "-o" else "white",
            capsize=2,
            lw=1.4,
            label=label_n(f"RFLO shared B, worst at dir {pole}", mask.sum()),
        )
    bptt, _ = D.profiles(ARM, "BPTT", 40000, dr.COMMON17)
    ax.errorbar(
        directions,
        bptt.mean(0),
        yerr=bptt.std(0) / np.sqrt(len(bptt)),
        fmt="-^",
        color=COLOR["BPTT"],
        capsize=2,
        lw=1.4,
        label=label_n("BPTT", len(bptt)),
    )
    direction_axis(ax)
    ax.set_ylabel("terminal error (fraction of reach)")
    ax.set_ylim(*ERROR_YLIM)
    ax.set_title("RFLO fails toward one pole of the 0-4 axis", loc="left")
    ax.legend(loc="upper center", fontsize=7, frameon=False)


# --- abstract panels ---------------------------------------------------------------------


def curve_matrix(D, run_ids):
    c = D.curves[D.curves.run_id.isin(run_ids)]
    table = c.pivot(index="step_end", columns="run_id", values="loss")
    return table.index.to_numpy(), table.to_numpy()


def draw_curves(ax, D, specs):
    for condition, budget, seeds, style in specs:
        ids = list(D.select(ARM, condition, budget, seeds).run_id)
        steps, L = curve_matrix(D, ids)
        k = steps / 1000
        ax.plot(k, L, color=COLOR[condition], lw=0.5, alpha=0.18, ls=style)
        ax.plot(
            k,
            L.mean(1),
            color=COLOR[condition],
            lw=1.8,
            ls=style,
            label=label_n(f"{NAME[condition]}, {budget // 1000}k", L.shape[1]),
        )
    ax.set_yscale("log")
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("training loss (1000-step mean)")
    ax.legend(loc="upper right", fontsize=7, frameon=False)


def draw_loss_40k(ax, D):
    draw_curves(
        ax,
        D,
        [(c, 40000, dr.COMMON17, "-") for c in ("BPTT", "projB0", "fixB", "varyB")],
    )
    ax.set_ylim(0.02, 0.5)
    ax.set_title("Training loss to 40k, COMMON17", loc="left")


def draw_loss_long(ax, D):
    draw_curves(
        ax,
        D,
        [
            ("BPTT", 100000, dr.LONG8, "-"),
            ("projB0", 100000, dr.LONG8, "-"),
            ("fixB", 100000, dr.LONG8, "-"),
            ("fixB", 300000, dr.LONG4, "--"),
        ],
    )
    ax.axvline(40, color=MUTED, lw=1, ls=":")
    ax.text(42, 0.017, "40k budget", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 300)
    ax.set_ylim(0.015, 0.5)
    ax.set_title("Longer budgets: no rule has flattened", loc="left")


BUDGETS = (40000, 100000, 300000)


def draw_error_by_budget(ax, D):
    """Mean error per seed at each budget a run exists for; lines join one seed."""
    offsets = {"BPTT": -0.22, "projB0": 0.0, "fixB": 0.22}
    plan = {
        "BPTT": [(40000, dr.LONG8), (100000, dr.LONG8)],
        "projB0": [(100000, dr.LONG8)],
        "fixB": [(40000, dr.LONG8), (100000, dr.LONG8), (300000, dr.LONG4)],
    }
    for condition, cells in plan.items():
        color = COLOR[condition]
        per_seed = {}
        for budget, seeds in cells:
            sel = D.select(ARM, condition, budget, seeds)
            sel = sel[sel.has_eval]
            x = BUDGETS.index(budget) + offsets[condition]
            for _, r in sel.iterrows():
                per_seed.setdefault(r.seed, []).append((x, r.mean_error))
            ax.scatter(
                np.full(len(sel), x), sel.mean_error, s=10, color=color, alpha=0.6
            )
            ax.plot(
                [x - 0.08, x + 0.08], [sel.mean_error.mean()] * 2, color=color, lw=2.5
            )
            ax.text(x, 0.338, f"n={len(sel)}", ha="center", fontsize=6.5, color=INK)
        for pts in per_seed.values():
            if len(pts) > 1:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, lw=0.6, alpha=0.4)
        ax.plot([], [], color=color, lw=2.5, label=NAME[condition])
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([f"{b // 1000}k" for b in BUDGETS])
    ax.set_xlabel("training budget (steps)")
    ax.set_ylabel("mean terminal error (fraction of reach)")
    ax.set_ylim(0, 0.35)
    ax.set_xlim(-0.5, 2.5)
    ax.text(
        0.0,
        0.01,
        "no projected run\nsaved at 40k",
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )
    ax.text(
        2 - 0.22,
        0.01,
        "no BPTT or\nprojected at 300k",
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )
    ax.set_title("Equal budgets, seeds 1 3 4 5 8-11", loc="left")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8), fontsize=7, frameon=False)


def draw_budget_profiles(axes, D):
    """fixB per-direction error per seed at 40k, 100k and 300k: the pole fades, then returns."""
    cells = [(40000, dr.LONG8), (100000, dr.LONG8), (300000, dr.LONG4)]
    for ax, (budget, seeds) in zip(axes, cells):
        P, _ = D.profiles(ARM, "fixB", budget, seeds)
        for row in P:
            ax.plot(
                range(dr.N_DIRECTIONS), row, color=COLOR["fixB"], lw=0.8, alpha=0.55
            )
            ax.scatter(row.argmax(), row.max(), s=12, color=COLOR["fixB"], zorder=3)
        ax.plot(range(dr.N_DIRECTIONS), P.mean(0), color=INK, lw=1.4, ls="--",
                label="mean")  # fmt: skip
        ax.set_ylim(*ERROR_YLIM)
        direction_axis(ax, compact=True)
        prefix = f"{NAME['fixB']}, " if ax is axes[0] else ""
        ax.set_title(f"{prefix}{budget // 1000}k (n={len(P)})", loc="left", fontsize=9)
    axes[0].set_ylabel("terminal error (fraction of reach)")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    for ax in axes[1:]:
        # Not set_yticklabels([]): under sharey that blanks the first panel's labels too.
        ax.tick_params(labelleft=False)
        ax.set_xlabel("")


def draw_multib(ax, D):
    """Worst direction for each net x B on the multiB grid, as an arrow on the target ring."""
    grid = dr.load_csv_summary("multib_grid.csv", D.dir)
    columns = ["B0", "B101", "B102", "B103", "own"]
    nets = dr.GRID_NETS
    angle = np.radians(dr.DIRECTION_DEG)
    for _, r in grid.iterrows():
        cx, cy = columns.index(r.B), -nets.index(r.net)
        ring = np.linspace(0, 2 * np.pi, 60)
        ax.plot(
            cx + 0.36 * np.cos(ring), cy + 0.36 * np.sin(ring), color="#cccccc", lw=0.6
        )
        ax.scatter(
            cx + 0.36 * np.cos(angle), cy + 0.36 * np.sin(angle), s=3, color="#aaaaaa"
        )
        a = angle[r.worst_dir]
        converged = bool(r.converged)
        ax.annotate(
            "",
            xy=(cx + 0.34 * np.cos(a), cy + 0.34 * np.sin(a)),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLOR["fixB"] if converged else MUTED,
                lw=1.6,
                ls="-" if converged else ":",
            ),
        )
        ax.text(cx + 0.4, cy + 0.25, str(r.worst_dir), fontsize=7, color=INK)
        if not converged:
            ax.text(cx, cy - 0.5, "not converged", ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(["B₀", "B 101", "B 102", "B 103", "own B"])
    ax.set_yticks([-i for i in range(len(nets))])
    ax.set_yticklabels([f"net {n}" for n in nets])
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(-len(nets) + 0.4, 0.6)
    ax.set_aspect("equal")
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Worst direction, RFLO at 40k: 0-4 axis for every B", loc="left")


def draw_pointmass(ax, D):
    """Point-mass per-direction error for both rules, on the arm panels' y scale."""
    directions = np.arange(dr.N_DIRECTIONS)
    for condition, marker in (("BPTT", "^"), ("fixB", "o")):
        P, _ = D.profiles(PM, condition, 70000)
        ax.errorbar(
            directions,
            P.mean(0),
            yerr=P.std(0) / np.sqrt(len(P)),
            fmt="-" + marker,
            color=COLOR[condition],
            capsize=2,
            lw=1.4,
            label=label_n(f"point mass, {NAME[condition]}", len(P)),
        )
    # Only BPTT's arm profile is a fair reference: RFLO's arm mean averages the two poles
    # into a double bump at dirs 0 and 4 that no single seed has.
    arm, _ = D.profiles(ARM, "BPTT", 40000, dr.COMMON17)
    ax.plot(directions, arm.mean(0), color=COLOR["BPTT"], lw=1, ls=":", alpha=0.8,
            label=label_n("arm 40k, BPTT (mean)", len(arm)))  # fmt: skip
    direction_axis(ax)
    ax.set_ylim(*ERROR_YLIM)
    ax.set_ylabel("terminal error (fraction of reach)")
    ax.set_title("Point-mass control: both rules matched", loc="left")
    ax.legend(loc="upper center", fontsize=7, frameon=False)


# --- assembly ----------------------------------------------------------------------------


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default=str(dr.DATA))
    p.add_argument("--seed", type=int, default=1, help="seed shown in the reach panel")
    args = p.parse_args()
    FIGURES.mkdir(exist_ok=True)
    D = Data(args.data)

    single = {
        "reaches": lambda ax: draw_reaches(ax, D, args.seed),
        "corridor": lambda ax: draw_corridor(ax, D),
        "profiles": lambda ax: draw_profiles(ax, D),
        "corridor_all": lambda ax: draw_corridor(ax, D, with_controls=True),
        "loss_40k": lambda ax: draw_loss_40k(ax, D),
        "loss_long": lambda ax: draw_loss_long(ax, D),
        "error_by_budget": lambda ax: draw_error_by_budget(ax, D),
        "multib_axis": lambda ax: draw_multib(ax, D),
        "pointmass": lambda ax: draw_pointmass(ax, D),
    }
    for name, draw in single.items():
        fig, ax = plt.subplots(figsize=(7.5 if name == "corridor_all" else 5.0, 4.2))
        draw(ax)
        save(fig, f"panel_{name}.png")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    draw_budget_profiles(axes, D)
    save(fig, "panel_budget_profiles.png")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, name in zip(axes, ("reaches", "corridor", "profiles")):
        single[name](ax)
    for ax, letter in zip(axes, "ABC"):
        ax.set_title(f"{letter}  " + ax.get_title(loc="left"), loc="left")
    fig.tight_layout()
    save(fig, "canalization_summary.png")

    fig = plt.figure(figsize=(17, 15))
    outer = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.28)
    layout = [
        ("reaches", outer[0, 0]),
        ("corridor_all", outer[0, 1]),
        ("profiles", outer[0, 2]),
        ("loss_40k", outer[1, 0]),
        ("loss_long", outer[1, 1]),
        ("error_by_budget", outer[1, 2]),
        ("multib_axis", outer[2, 0]),
        ("budget_profiles", outer[2, 1]),
        ("pointmass", outer[2, 2]),
    ]
    for (name, cell), letter in zip(layout, "ABCDEFGHI"):
        if name == "budget_profiles":
            inner = cell.subgridspec(1, 3, wspace=0.08)
            axes = [fig.add_subplot(inner[0, i]) for i in range(3)]
            draw_budget_profiles(axes, D)
            axes[0].set_title(
                f"{letter}  " + axes[0].get_title(loc="left"), loc="left", fontsize=9
            )
            continue
        ax = fig.add_subplot(cell)
        single[name](ax)
        ax.set_title(f"{letter}  " + ax.get_title(loc="left"), loc="left")
    save(fig, "abstract_summary.png")
    print(f"wrote {FIGURES}/canalization_summary.png, abstract_summary.png and panels")


if __name__ == "__main__":
    main()
