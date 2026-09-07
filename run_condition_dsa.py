"""Activation DSA grouped by feedback condition (fixB vs varyB), not by rule.

The E-varyB test: does *sharing* one fixed feedback matrix B pull independently-
initialized RFLO networks into a tighter region of function space than giving each
its own B? This is the primary, gauge-invariant readout for that question -- the
same activation-DSA machinery used for the rule contrast (run_activation_dsa_lean),
with the grouping variable swapped from learning rule to feedback condition. It is
NOT a new metric; summarize_within_between takes an arbitrary grouping via `rules=`.

  within-fixB  : across-seed DSA distance among RFLO seeds that SHARE B (b_seed=0)
  within-varyB : across-seed DSA distance among RFLO seeds with their OWN B (match)
  prediction   : within-fixB < within-varyB  (shared B => tighter cluster)

Loads one artifact at a time (H only) to stay under memory, like the lean rule
runner. Convergence-gated first: an undertrained net has no stable trajectory to
place in the cluster (cf. the non-converged Bonly_b3 run), so seeds whose loss has
not flattened are dropped, and only seeds converged in BOTH conditions are kept.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

import analysis


def converged(losses, window=2000, tol=2e-3):
    L = np.asarray(losses, dtype=float)
    w = np.array([L[i : i + window].mean() for i in range(0, len(L) - window + 1, window)])
    return len(w) >= 2 and (w[-2] - w[-1]) < tol


def load_H(results_dir, effector, seed):
    """RFLO hidden states + direction index for one seed, plus its convergence flag."""
    path = Path(results_dir) / f"{effector}_seed{seed}_RFLO.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    lean = {"H": data["H"], "direction_idx": data["direction_idx"]}
    ok = converged(data["losses"])
    del data
    gc.collect()
    return lean, ok


def main():
    p = argparse.ArgumentParser(description="Condition-grouped (fixB vs varyB) activation DSA.")
    p.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"])
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--fix-dir", default="results/evaryB_fixB")
    p.add_argument("--vary-dir", default="results/evaryB_varyB")
    p.add_argument("--n-delays", type=int, default=10)
    p.add_argument("--rank", type=int, default=20)
    args = p.parse_args()

    conditions = [("fixB", args.fix_dir), ("varyB", args.vary_dir)]

    # Convergence gate: keep only seeds converged in BOTH conditions.
    kept, dropped = [], []
    for s in args.seeds:
        oks = {}
        for cond, d in conditions:
            try:
                _, ok = load_H(d, args.effector, s)
                oks[cond] = ok
            except FileNotFoundError:
                oks[cond] = None
        if all(oks[c] for c, _ in conditions):
            kept.append(s)
        else:
            dropped.append((s, oks))
    for s, oks in dropped:
        print(f"Dropped seed{s} (not converged/missing in both): {oks}")
    if len(kept) < 2:
        print(f"Only {len(kept)} seeds converged in both conditions -- need >=2. Stopping.")
        return
    print(f"Seeds in both conditions, converged (n={len(kept)}): {kept}")

    systems, labels = [], []
    for cond, d in conditions:
        for s in kept:
            run, _ = load_H(d, args.effector, s)
            systems.append(analysis.extract_per_direction_trajectories(run))
            labels.append(f"{cond}-{s}")  # prefix is the grouping key for summarize_within_between
            del run
            gc.collect()

    print(f"Running activation DSA on {len(systems)} systems "
          f"(n_delays={args.n_delays}, rank={args.rank})...")
    similarities = analysis.run_dsa(systems, n_delays=args.n_delays, rank=args.rank)

    summary = analysis.summarize_within_between(similarities, labels)
    for pair, mean_dist in summary.items():
        print(f"{pair}: mean DSA distance = {mean_dist:.4f}")

    p_val = analysis.calculate_p_value_of_dsa_distance(
        similarities, labels, rule_kinds=("fixB", "varyB")
    )
    # Free/unpaired permutation, matching run_activation_dsa_lean; the block/paired
    # version (permute condition within seed) is the roadmap statistics fix.
    print(f"Label-permutation p-value (free, unpaired): {p_val:.5f}")

    out = Path(args.fix_dir).parent / f"{args.effector}_condition_dsa_matrix.npy"
    np.save(out, similarities)
    # Sidecar records the exact row order (condition-major over the kept seeds) so
    # the paired-permutation applier pairs rows without re-deriving the gate.
    labels_out = out.with_name(out.name[: -len("_matrix.npy")] + "_labels.json")
    labels_out.write_text(json.dumps(labels))
    print(f"Saved DSA matrix to: {out}")
    print(f"Saved row-order labels to: {labels_out}")
    print("Prediction: within-(fixB,fixB) < within-(varyB,varyB) => shared B canalizes.")


if __name__ == "__main__":
    main()
