"""Report training-loss convergence for saved run artifacts.

A run enters a cross-condition comparison only if it has both flattened and landed.
Flattening alone is not enough: a run can stall in a failed high basin, look perfectly
converged by the drop criterion, and still never have learned the task, which would
enter a dispersion metric as a far-flung outlier for reasons having nothing to do with
the manipulation. Hence the absolute ``--loss-gate`` alongside the drop test. This is
the gate that defines the seed rosters used everywhere downstream.
"""

import argparse
import glob
import os
import re

import torch

from analysis import loss_windows


def main():
    p = argparse.ArgumentParser(description="Convergence check for run artifacts.")
    p.add_argument("--dir", required=True, help="Directory of .pt artifacts")
    p.add_argument("--rule", default="RFLO", choices=["RFLO", "BPTT"])
    p.add_argument(
        "--window", type=int, default=2000, help="Steps per averaging window"
    )
    p.add_argument(
        "--tol",
        type=float,
        default=2e-3,
        help="Flattened if the final window's drop from the previous one is below this",
    )
    p.add_argument(
        "--loss-gate",
        type=float,
        default=0.10,
        help="Landed if the final window's mean loss is below this",
    )
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, f"*_{args.rule}.pt")))
    if not files:
        print(f"No *_{args.rule}.pt artifacts in {args.dir}")
        return

    print(
        f"Convergence check: {args.dir}  (rule={args.rule}, window={args.window}, "
        f"tol={args.tol}, loss-gate={args.loss_gate})"
    )
    good, excluded = [], []
    for f in files:
        a = torch.load(f, map_location="cpu", weights_only=False)
        w = loss_windows(a["losses"], args.window)
        name = os.path.basename(f)
        if len(w) < 2:
            print(f"{name}: too few steps for a window of {args.window}")
            continue
        drop = w[-2] - w[-1]
        flattened, landed = drop < args.tol, w[-1] < args.loss_gate
        if not flattened:
            verdict = "STILL DESCENDING"
        elif not landed:
            # Flat but high: stalled in a failed basin, not converged on the task.
            verdict = "FAILED BASIN"
        else:
            verdict = "converged"

        seed_match = re.search(r"_seed(\d+)_", name)
        seed = int(seed_match.group(1)) if seed_match else None
        if seed is not None:
            (good if verdict == "converged" else excluded).append((seed, w[-1]))

        idx = [0, len(w) // 2, len(w) - 1]
        snap = "  ".join(f"{w[i]:.4f}" for i in idx)
        lr = a.get("lr")
        steps = a.get("num_steps")
        print(
            f"{name:32s} lr={lr} steps={steps} "
            f"loss[0,50,100%]={snap}  last-drop={drop:+.5f}  -> {verdict}"
        )

    if good or excluded:
        # Printed as a seed list so the passing roster can be pasted straight into
        # the analysis scripts, which must run on one roster shared by all conditions.
        n = len(good) + len(excluded)
        print(f"\nGOOD {len(good)}/{n}: {' '.join(str(s) for s, _ in sorted(good))}")
        excl = ", ".join(f"{s} ({loss:.3f})" for s, loss in sorted(excluded))
        print(f"EXCL {len(excluded)}/{n}: {excl if excl else '-'}")


if __name__ == "__main__":
    main()
