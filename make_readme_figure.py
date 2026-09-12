"""Build the three-panel summary figure for the README.

Panel A  both rules drive the arm to all eight targets, RFLO with larger errors.
Panel B  learning subspaces of RFLO seeds sharing one feedback matrix are far from
         orthogonal; BPTT's and per-seed-B RFLO's sit at the random-subspace null.
Panel C  RFLO's per-direction deficit is bimodal: two opposing arcs, roughly half the
         seeds each, against BPTT's flat profile.

Panels A and C read run artifacts; panel B reads the distance matrices written by
run_subspace_geometry.py. The reach profiles are cached in RESULTS/PROFILE_CACHE,
since collecting them touches every weight-history artifact (~0.66 GB each).

Usage:
    python make_readme_figure.py [--refresh]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import analysis

RESULTS = Path("results")
FIGURES = Path("figures")
PROFILE_CACHE = RESULTS / "arm_reach_profiles.npz"

EFFECTOR = "RigidTendonArm26"
# Seeds that converged in all three arm conditions (see check_convergence.py).
COMMON17 = [1, 3, 4, 5, 8, 9, 10, 11, 12, 14, 18, 19, 20, 21, 22, 23, 24]
CONDITIONS = {
    "fixB": (RESULTS / "evaryB_fixB_wh", "RFLO"),
    "varyB": (RESULTS / "evaryB_varyB_wh", "RFLO"),
    "BPTT": (RESULTS / "arm_bptt_40k_wh", "BPTT"),
}
K = 3  # learning-subspace dimension used for the published contrast

RFLO_COLOR = "tab:orange"
BPTT_COLOR = "tab:blue"
VARYB_COLOR = "tab:gray"


def artifact(condition, seed):
    directory, rule = CONDITIONS[condition]
    return directory / f"{EFFECTOR}_seed{seed}_{rule}.pt"


def reach_profiles(refresh=False):
    """Per-condition (n_seeds, n_directions) reach-relative terminal error."""
    if PROFILE_CACHE.exists() and not refresh:
        return dict(np.load(PROFILE_CACHE))
    scale = analysis.reach_distance(EFFECTOR)
    profiles = {}
    for condition in CONDITIONS:
        rows = []
        for seed in COMMON17:
            run = torch.load(
                artifact(condition, seed), map_location="cpu", weights_only=False
            )
            FT, goal = run["FT"].numpy(), run["goal"].numpy()
            rows.append(np.linalg.norm(FT[-1] - goal, axis=-1) / scale)
            del run
        profiles[condition] = np.stack(rows)
    np.savez(PROFILE_CACHE, **profiles)
    return profiles


def arc_subtypes(profiles):
    """Split seeds by the sign of their first principal-component score.

    The deficit is bimodal, so a mean over seeds averages two anti-correlated shapes
    into a double bump no individual seed has. PC1 of the centered profiles is the arc
    axis, and its sign labels which arc a seed fails on.
    """
    centered = profiles - profiles.mean(axis=0)
    pc1 = np.linalg.svd(centered, full_matrices=False)[2][0]
    scores = centered @ pc1
    # Orient the axis so a positive score means the later directions are the bad ones.
    if np.argmax(pc1) < len(pc1) / 2:
        scores = -scores
    return scores >= 0


def within_angles():
    """Mean within-condition principal angle, in degrees, plus the random null."""
    angles = {}
    for stem in (
        f"{EFFECTOR}_subspace_varyB_vs_fixB",
        f"{EFFECTOR}_subspace_BPTT_vs_fixB",
    ):
        path = RESULTS / f"{stem}_geodesic_k{K}"
        M = np.load(path.with_name(path.name + "_matrix.npy"))
        labels = json.loads(path.with_name(path.name + "_labels.json").read_text())
        summary = analysis.summarize_within_between(M, labels)
        for (a, b), distance in summary.items():
            if a == b:
                angles[a] = np.degrees(distance / np.sqrt(K))
    null_mean, _ = analysis.random_subspace_distance(64 * 64, K, metric="geodesic")
    return angles, np.degrees(null_mean / np.sqrt(K))


def draw_reaches(ax, seed=1):
    """Fingertip paths for one seed under each rule."""
    for condition, color in (("BPTT", BPTT_COLOR), ("fixB", RFLO_COLOR)):
        run = torch.load(
            artifact(condition, seed), map_location="cpu", weights_only=False
        )
        FT, goal = run["FT"].numpy(), run["goal"].numpy()
        label = "BPTT" if condition == "BPTT" else "RFLO"
        for d in range(FT.shape[1]):
            ax.plot(
                FT[:, d, 0],
                FT[:, d, 1],
                color=color,
                lw=1.4,
                alpha=0.85,
                label=label if d == 0 else None,
            )
        if condition == "fixB":
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
            ax.scatter(FT[0, 0, 0], FT[0, 0, 1], color="black", s=25, zorder=6)
        del run
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("hand position x (m)")
    ax.set_ylabel("hand position y (m)")
    ax.set_title(f"A  Reaching behavior, seed {seed}", loc="left")
    ax.legend(loc="best", fontsize=8, frameon=False)


def draw_corridor(ax, angles, null):
    """Within-condition learning-subspace angle against the random-subspace null."""
    order = [
        ("RFLO\nshared B", "fixB", RFLO_COLOR),
        ("BPTT", "BPTT", BPTT_COLOR),
        ("RFLO\nper-seed B", "varyB", VARYB_COLOR),
    ]
    values = [angles[key] for _, key, _ in order]
    ax.bar(
        [name for name, _, _ in order],
        values,
        color=[color for _, _, color in order],
        width=0.6,
    )
    for i, v in enumerate(values):
        ax.text(i, v + 1.0, f"{v:.1f}°", ha="center", va="bottom", fontsize=9)
    ax.axhline(null, ls="--", color="black", lw=1)
    ax.text(
        -0.45, null + 1.5, "independent subspaces", ha="left", va="bottom", fontsize=8
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("angle between seeds' learning subspaces")
    ax.set_title("B  Sharing B aligns the weight updates", loc="left")


def draw_profiles(ax, profiles):
    """Per-direction error: the two RFLO arcs against BPTT's flat profile."""
    fixB = profiles["fixB"]
    late = arc_subtypes(fixB)
    directions = np.arange(fixB.shape[1])
    for mask, style, name in (
        (late, "-o", "RFLO, arc 3-5"),
        (~late, "-s", "RFLO, arc 0-2"),
    ):
        group = fixB[mask]
        mean, sem = group.mean(0), group.std(0) / np.sqrt(len(group))
        ax.errorbar(
            directions,
            mean,
            yerr=sem,
            fmt=style,
            color=RFLO_COLOR,
            markerfacecolor="white" if name.endswith("7-0-1") else RFLO_COLOR,
            capsize=2,
            lw=1.4,
            label=f"{name} (n={mask.sum()})",
        )
    bptt = profiles["BPTT"]
    ax.errorbar(
        directions,
        bptt.mean(0),
        yerr=bptt.std(0) / np.sqrt(len(bptt)),
        fmt="-^",
        color=BPTT_COLOR,
        capsize=2,
        lw=1.4,
        label=f"BPTT (n={len(bptt)})",
    )
    ax.set_xticks(directions)
    ax.set_xlabel("target direction")
    ax.set_ylabel("terminal error (fraction of reach)")
    ax.set_ylim(0, 0.62)
    ax.set_title("C  RFLO fails one arc, not all directions", loc="left")
    ax.legend(loc="upper center", ncol=2, fontsize=7.5, frameon=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--refresh", action="store_true", help="Recompute the reach-profile cache"
    )
    args = p.parse_args()

    FIGURES.mkdir(exist_ok=True)
    profiles = reach_profiles(refresh=args.refresh)
    angles, null = within_angles()
    print(
        "within-condition angle (deg): "
        + "  ".join(f"{k} {v:.1f}" for k, v in sorted(angles.items()))
        + f"  | random null {null:.1f}"
    )
    for condition, P in profiles.items():
        print(
            f"{condition}: mean error {P.mean():.3f} of reach, worst direction {P.mean(0).argmax()}"
        )

    panels = [
        ("reaches", draw_reaches, ()),
        ("corridor", draw_corridor, (angles, null)),
        ("profiles", draw_profiles, (profiles,)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (_, draw, extra) in zip(axes, panels):
        draw(ax, *extra)
    fig.tight_layout()
    fig.savefig(FIGURES / "canalization_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for name, draw, extra in panels:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        draw(ax, *extra)
        fig.tight_layout()
        fig.savefig(FIGURES / f"panel_{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {FIGURES}/canalization_summary.png and three standalone panels")


if __name__ == "__main__":
    main()
