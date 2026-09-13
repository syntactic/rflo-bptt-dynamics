"""Per-direction reach-relative terminal error for saved run artifacts.

The aggregate error hides the shape of a failure: missing every direction a little
looks, on the mean, like nailing most and collapsing on a few. Printing the per-seed
breakdown also separates a deficit that repeats across independently initialized seeds
from one that is idiosyncratic. Error is divided by the effector's reach distance, so
directions and effectors sit on one scale.
"""

import argparse

import numpy as np
import torch

from analysis import reach_relative_terminal_error


def per_direction_rel_error(path):
    """(n_targets,) reach-relative terminal error for one saved run."""
    d = torch.load(path, map_location="cpu")
    FT = d["FT"].detach().cpu().numpy()
    # Artifacts store FT time-first; the library metric wants batch-first.
    FT = np.transpose(FT, (1, 0, 2))
    return reach_relative_terminal_error(FT, d["goal"], d["effector"])


def main():
    p = argparse.ArgumentParser(
        description="Per-direction reach-relative terminal error."
    )
    p.add_argument("--dir", required=True, help="Directory of .pt artifacts")
    p.add_argument("--effector", default="RigidTendonArm26")
    p.add_argument("--rules", nargs="+", default=["BPTT", "RFLO"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--per-seed", action="store_true", help="Also print per-seed rows")
    args = p.parse_args()

    print(f"Per-direction reach-relative terminal error: {args.dir} ({args.effector})")
    for rule in args.rules:
        rows = []
        for s in args.seeds:
            path = f"{args.dir}/{args.effector}_seed{s}_{rule}.pt"
            try:
                rows.append((s, per_direction_rel_error(path)))
            except FileNotFoundError:
                continue
        if not rows:
            print(f"\n{rule}: (no artifacts)")
            continue
        stk = np.stack([r for _, r in rows])
        m = stk.mean(0)
        ndir = m.shape[0]
        header = " ".join(f"{i:5d}" for i in range(ndir))
        print(f"\n{rule}  (n={len(rows)})   dir: {header}")
        print(
            "  mean over seeds:  "
            + " ".join(f"{v:5.2f}" for v in m)
            + f"   | mean {m.mean():.3f}  worst dir {int(m.argmax())} ({m.max():.2f})"
        )
        if args.per_seed:
            for s, r in rows:
                print(f"  seed {s:<11d}" + " ".join(f"{v:5.2f}" for v in r))


if __name__ == "__main__":
    main()
