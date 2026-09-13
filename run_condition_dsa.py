"""Activation DSA grouped by feedback condition (fixB / varyB / aligned / BPTT).

Tests whether the feedback matrix leaves a signature in function space, the
gauge-invariant channel. All requested conditions go into ONE matrix so every distance
shares a delay embedding and rank, and so the BPTT reference is on the same roster
rather than borrowed from another run. Each system's operator is fitted independently,
so a two-condition block of the matrix is the comparison that pair would produce alone
(agreement with separate runs ~2e-4).

Loads artifacts one at a time: weight-history artifacts are ~0.66 GB each and the
roster does not fit in memory at once. Seeds that did not both flatten and land are
dropped, since an unsettled run fakes within-group dispersion.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

import analysis


def load_H(results_dir, effector, seed, rule="RFLO"):
    """Hidden states + direction index for one seed, plus its convergence flag."""
    path = Path(results_dir) / f"{effector}_seed{seed}_{rule}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    lean = {"H": data["H"], "direction_idx": data["direction_idx"]}
    ok = analysis.converged(data["losses"])
    del data
    gc.collect()
    return lean, ok


def main():
    p = argparse.ArgumentParser(
        description="Condition-grouped (fixB / varyB / aligned) activation DSA."
    )
    p.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"])
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--fix-dir", default="results/evaryB_fixB")
    p.add_argument("--vary-dir", default="results/evaryB_varyB")
    p.add_argument(
        "--aligned-dir",
        default=None,
        help="readout-aligned condition directory; omit to run fixB vs varyB only",
    )
    p.add_argument(
        "--bptt-dir",
        default=None,
        help="BPTT reference directory; omit to leave the exact-gradient anchor out",
    )
    p.add_argument("--n-delays", type=int, default=10)
    p.add_argument("--rank", type=int, default=20)
    p.add_argument(
        "--out-dir",
        default=None,
        help="where the matrix and its label sidecar are written "
        "(default: the parent of --fix-dir)",
    )
    args = p.parse_args()

    conditions = [
        ("fixB", args.fix_dir, "RFLO"),
        ("varyB", args.vary_dir, "RFLO"),
    ]
    if args.aligned_dir:
        conditions.append(("aligned", args.aligned_dir, "RFLO"))
    if args.bptt_dir:
        conditions.append(("BPTT", args.bptt_dir, "BPTT"))

    # Keep only seeds that converged in EVERY condition: the paired permutation
    # needs each seed to carry all condition labels, and a roster that differs by
    # condition would make the within-group dispersions incomparable.
    kept, dropped = [], []
    for s in args.seeds:
        oks = {}
        for cond, d, rule in conditions:
            try:
                _, ok = load_H(d, args.effector, s, rule)
                oks[cond] = ok
            except FileNotFoundError:
                oks[cond] = None
        if all(oks[c] for c, _, _ in conditions):
            kept.append(s)
        else:
            dropped.append((s, oks))
    for s, oks in dropped:
        print(f"Dropped seed{s} (not converged/missing in all conditions): {oks}")
    if len(kept) < 2:
        print(
            f"Only {len(kept)} seeds converged in every condition, need >=2. Stopping."
        )
        return
    print(
        f"Conditions: {[c for c, _, _ in conditions]}\n"
        f"Seeds converged in all conditions (n={len(kept)}): {kept}"
    )

    systems, labels = [], []
    for cond, d, rule in conditions:
        for s in kept:
            run, _ = load_H(d, args.effector, s, rule)
            systems.append(analysis.extract_per_direction_trajectories(run))
            # The condition prefix is the grouping key summarize_within_between reads
            labels.append(f"{cond}-{s}")
            del run
            gc.collect()

    print(
        f"Running activation DSA on {len(systems)} systems "
        f"(n_delays={args.n_delays}, rank={args.rank})..."
    )
    similarities = analysis.run_dsa(systems, n_delays=args.n_delays, rank=args.rank)

    summary = analysis.summarize_within_between(similarities, labels)
    for pair, mean_dist in sorted(summary.items()):
        kind = "within" if pair[0] == pair[1] else "between"
        name = pair[0] if pair[0] == pair[1] else f"{pair[0]}/{pair[1]}"
        print(f"{kind}-{name}: mean DSA distance = {mean_dist:.4f}")

    # Runs share network init and target streams across conditions, violating free
    # exchangeability. Persist the distance matrix here and run paired/block
    # permutations with run_paired_permutation.py, which subsets the two groups of
    # each contrast out of this matrix.
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.fix_dir).parent
    tag = "-".join(c for c, _, _ in conditions)
    out = out_dir / f"{args.effector}_condition_dsa_{tag}_matrix.npy"
    np.save(out, similarities)
    # Save row labels so paired permutation testing matches seed pairs directly
    labels_out = out.with_name(out.name[: -len("_matrix.npy")] + "_labels.json")
    labels_out.write_text(json.dumps(labels))
    print(f"Saved DSA matrix to: {out}")
    print(f"Saved row-order labels to: {labels_out}")
    print("Next: run run_paired_permutation.py on this matrix for paired inference.")


if __name__ == "__main__":
    main()
