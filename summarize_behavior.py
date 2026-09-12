"""Report the behavioral gap between BPTT and RFLO from saved artifacts.

This is the "behavior-matching gate" readout: before any
Q4 canalization analysis, we need to know whether the two rules reach matched behavior.

The primary metric is the continuous, reach-relative terminal error
(`terminal_error / reach_distance`), which puts the point mass (reach 0.5) and arm
(reach 0.1) on the same scale. Binary success is a secondary readability aid, shown
at two fixed reach fractions (10% and 5%); it is recomputed here from stored FT/goal at a
reach-relative bar, not read from the artifact's training-time `metrics` (which used an
unscaled absolute 0.05 radius, see analysis.reach_metrics).

The RFLO-BPTT gap in mean reach-relative terminal error is what the TOST equivalence test
(Delta = 5% of reach, frozen 2026-09-01) will adjudicate at N=15. At small n this is a
direction only, not an equivalence verdict.

Usage:
    python summarize_behavior.py --dir results/clean --n-seeds 15 --per-seed
"""

import argparse
import os

import numpy as np
import torch

from analysis import (
    reach_distance,
    reach_metrics,
    reach_relative_terminal_error,
)

# Frozen behavior-matching criteria (set before inspecting any BPTT-vs-RFLO outcome).
TOST_MARGIN_FRAC = 0.05  # Delta for the equivalence test, as a fraction of reach.
SUCCESS_FRACS = (
    0.10,
    0.05,
)  # binary success bars, as fractions of reach (readability only)


def _load_ft_goal(d):
    """Extract batch-first FT (n_targets, T, 2) and goal (n_targets, 2) from an artifact,
    mirroring diagnose_endpoints.py: FT is stored (T, B, 2) by inference()."""
    FT = (
        d["FT"].detach().cpu().numpy()
        if torch.is_tensor(d["FT"])
        else np.asarray(d["FT"])
    )
    goal = (
        d["goal"].detach().cpu().numpy()
        if torch.is_tensor(d["goal"])
        else np.asarray(d["goal"])
    )
    goal = goal.reshape(goal.shape[0], -1)[:, :2]
    if FT.shape[0] != goal.shape[0]:  # (T, B, 2) -> (B, T, 2)
        FT = np.transpose(FT, (1, 0, 2))
    return FT, goal


def cell_stats(results_dir, effector, rule, n_seeds):
    """Per-seed behavioral metrics for one (effector, rule).

    Returns a dict of equal-length arrays: final_loss (tail-averaged training loss),
    rel_terr (mean reach-relative terminal error over the 8 targets), succ10/succ05
    (binary success at 10%/5% of reach), path_dev (reach-relative path deviation).
    """
    reach = reach_distance(effector)
    finals, rel_terr, succ10, succ05, dev = [], [], [], [], []
    for s in range(n_seeds):
        f = os.path.join(results_dir, f"{effector}_seed{s}_{rule}.pt")
        if not os.path.exists(f):
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        finals.append(float(np.mean(d["losses"][-50:])))  # tail-averaged loss
        FT, goal = _load_ft_goal(d)
        rel_terr.append(float(reach_relative_terminal_error(FT, goal, effector).mean()))
        # Reuse reach_metrics with a reach-scaled bar so success is comparable across effectors.
        succ10.append(
            reach_metrics(FT, goal, success_radius=SUCCESS_FRACS[0] * reach)[0]
        )
        succ05.append(
            reach_metrics(FT, goal, success_radius=SUCCESS_FRACS[1] * reach)[0]
        )
        # Path deviation is also in effector units, so scale it too for cross-effector reads.
        dev.append(reach_metrics(FT, goal)[1] / reach)
    return {
        "final": np.array(finals),
        "rel_terr": np.array(rel_terr),
        "succ10": np.array(succ10),
        "succ05": np.array(succ05),
        "dev": np.array(dev),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/clean", help="artifact directory")
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument(
        "--effectors", nargs="+", default=["ReluPointMass24", "RigidTendonArm26"]
    )
    ap.add_argument(
        "--per-seed",
        action="store_true",
        help="Also print per-seed rows",
    )
    args = ap.parse_args()

    print(f"Behavioral summary for {args.dir} (up to {args.n_seeds} seeds)")
    print(
        "Primary metric = reach-relative terminal error (mean+-sd over seeds). "
        f"success bars: {SUCCESS_FRACS[0]:.0%}/{SUCCESS_FRACS[1]:.0%} of reach (readability).\n"
    )
    header = (
        f"{'effector':18s} {'rule':4s} {'n':>2s}  {'final_loss':>10s}  "
        f"{'rel_terr':>16s}  {'succ@10%':>8s}  {'succ@5%':>8s}  {'path_dev':>8s}"
    )
    print(header)
    print("-" * len(header))
    for eff in args.effectors:
        cells = {}
        for rule in ["BPTT", "RFLO"]:
            c = cell_stats(args.dir, eff, rule, args.n_seeds)
            cells[rule] = c
            n = len(c["final"])
            if n == 0:
                print(f"{eff:18s} {rule:4s}  0  (no artifacts)")
                continue
            rt = c["rel_terr"]
            print(
                f"{eff:18s} {rule:4s} {n:2d}  {c['final'].mean():10.4f}  "
                f"{f'{rt.mean():.3f}+-{rt.std():.3f}':>16s}  "
                f"{c['succ10'].mean():8.3f}  {c['succ05'].mean():8.3f}  {c['dev'].mean():8.3f}"
            )
            if args.per_seed:
                for i in range(n):
                    print(
                        f"{'  seed ' + str(i):18s} {rule:4s}    {c['final'][i]:10.4f}  "
                        f"{c['rel_terr'][i]:16.3f}  {c['succ10'][i]:8.3f}  "
                        f"{c['succ05'][i]:8.3f}  {c['dev'][i]:8.3f}"
                    )
        if len(cells["BPTT"]["final"]) and len(cells["RFLO"]["final"]):
            gap = cells["RFLO"]["rel_terr"].mean() - cells["BPTT"]["rel_terr"].mean()
            n_min = min(len(cells["BPTT"]["final"]), len(cells["RFLO"]["final"]))
            verdict = "within" if abs(gap) <= TOST_MARGIN_FRAC else "outside"
            tag = "" if n_min >= 15 else "  [direction only, n<15, no TOST verdict]"
            print(
                f"{'':18s} gap  ->  RFLO-BPTT rel_terr {gap:+.3f}  "
                f"({verdict} Delta={TOST_MARGIN_FRAC:.2f}){tag}\n"
            )


if __name__ == "__main__":
    main()
