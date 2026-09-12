"""Weight-channel learning-subspace (Grassmann) geometry across feedback conditions.

Each seed's recurrent-weight trajectory is centered on W(0), coarse-grained, and reduced to
its top-k right singular vectors: the directions learning actually moved along. Two seeds are
compared by the principal angles between those subspaces, which depend only on the subspaces
themselves and not on the arbitrary basis an SVD returns for either one. A condition's
dispersion is its mean within-group distance, and canalization means one condition's seeds sit
closer together than another's.

Inference is the paired (block) permutation. A seed's runs in two conditions share their
initialization and target stream, so the exchangeable unit is the seed and the null flips the
condition label within each seed. Every p-value from one invocation enters a single
Holm-Bonferroni family.

The distances mean nothing on their own. Independent k-subspaces of 4096 dimensions are
near-orthogonal (~88.7 deg at k=3), so a within-group angle reads as a corridor only when it
sits well below that anchor, which is printed alongside every contrast.

Gauge caveat: independently initialized seeds label their hidden units differently, so raw
cross-seed weight geometry mixes real structure with relabeling. The defensible frame is the
shared-init paired one, which is what a contrast between two conditions of the same seed set
compares.

Example:
    python run_subspace_geometry.py RigidTendonArm26 \
        --condition fixB=results/evaryB_fixB_wh=RFLO \
        --condition varyB=results/evaryB_varyB_wh=RFLO \
        --condition BPTT=results/arm_bptt_40k_wh \
        --contrast varyB-vs-fixB --contrast BPTT-vs-fixB \
        --seeds 1 3 4 5 8 9 10 11 12 14 18 19 20 21 22 23 24
"""

import argparse
import gc
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

import analysis

# One group-token grammar and one results table across the project's permutation reports.
from run_paired_permutation import format_table, parse_group


def parse_contrast(token, known):
    """Parse 'A-vs-B' into (baseline, hypothesized-tighter) condition labels.

    Order matters: paired_permutation_test scores within(first) - within(second), so a
    positive observed statistic means the second condition clusters tighter.
    """
    parts = token.split("-vs-")
    if len(parts) != 2:
        raise SystemExit(f"contrast '{token}' must look like 'varyB-vs-fixB'")
    unknown = [p for p in parts if p not in known]
    if unknown:
        raise SystemExit(
            f"contrast '{token}' names undeclared condition(s) {unknown}; "
            f"declared: {sorted(known)}"
        )
    return tuple(parts)


def subspace_bases(condition, effector, seeds, k, bin_size):
    """Per-seed (d, k) bases for one condition, with the mean variance the k covers.

    Weight-history artifacts run ~0.66 GB each, so every one is loaded, reduced to its basis,
    and freed before the next: at 17 seeds the histories would not fit in memory together.
    """
    bases, covered, missing = {}, [], []
    for seed in seeds:
        path = (
            Path(condition.directory) / f"{effector}_seed{seed}_{condition.suffix}.pt"
        )
        if not path.exists():
            missing.append(seed)
            continue
        run = torch.load(path, map_location="cpu", weights_only=False)
        basis, cumvar = analysis.learning_subspace(
            run, k, bin_size=bin_size, return_spectrum=True
        )
        bases[seed] = basis
        covered.append(cumvar[basis.shape[1] - 1])
        del run
        gc.collect()
    return bases, covered, missing


def distance_matrix(bases_by_label, labels, metric):
    """Pairwise Grassmann distances, in the row order of `labels`."""
    n = len(labels)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            theta = analysis.principal_angles(
                bases_by_label[labels[i]], bases_by_label[labels[j]]
            )
            M[i, j] = M[j, i] = analysis.grassmann_distance(theta, metric)
    return M


def typical_angle(distance, k, metric):
    """Geodesic distance as a per-direction angle in degrees; None for other metrics.

    A geodesic distance is ||theta||, which grows as sqrt(k), so dividing by sqrt(k) recovers
    the root-mean-square principal angle and stays comparable across k. The chordal metric
    has no such reading because it measures ||sin theta||.
    """
    if metric != "geodesic":
        return None
    return np.degrees(distance / np.sqrt(k))


def run_contrast(contrast, bases, seeds, k, metric, save_dir, effector):
    """Within/between dispersion and the paired test for one contrast and metric."""
    baseline, tighter = contrast
    paired_seeds = [s for s in seeds if s in bases[baseline] and s in bases[tighter]]
    if len(paired_seeds) < 2:
        print(f"  [skip] {baseline}-vs-{tighter}: {len(paired_seeds)} paired seeds")
        return None

    labels = [f"{cond}-{s}" for cond in (baseline, tighter) for s in paired_seeds]
    by_label = {
        f"{cond}-{s}": bases[cond][s] for cond in contrast for s in paired_seeds
    }
    M = distance_matrix(by_label, labels, metric)

    summary = analysis.summarize_within_between(M, labels)
    result = analysis.paired_permutation_test(M, labels, rule_kinds=contrast)
    within = {c: summary[(c, c)] for c in contrast}
    between = summary[tuple(sorted(contrast))]

    stem = Path(save_dir) / f"{effector}_subspace_{baseline}_vs_{tighter}_{metric}_k{k}"
    np.save(stem.with_name(stem.name + "_matrix.npy"), M)
    stem.with_name(stem.name + "_labels.json").write_text(json.dumps(labels))

    return {
        "cell": f"{effector}:{baseline}-vs-{tighter}",
        "metric": f"subspace_{metric}_k{k}",
        "observed": result["observed"],
        "p_value": result["p_value"],
        "n_perms": result["n_permutations"],
        "n_pairs": len(paired_seeds),
        "within": within,
        "between": between,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("effector", choices=["ReluPointMass24", "RigidTendonArm26"])
    p.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="LABEL=DIR[=SUFFIX]",
        help="a condition: its label, artifact directory, and rule suffix "
        "(defaults to the label). Repeatable.",
    )
    p.add_argument(
        "--contrast",
        action="append",
        required=True,
        metavar="A-vs-B",
        help="a paired contrast between two declared conditions, baseline first; "
        "positive observed means B is tighter. Repeatable.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=15)
    p.add_argument(
        "--k", type=int, default=3, help="learning-subspace dimension (default: 3)"
    )
    p.add_argument("--bin-size", type=int, default=50)
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["geodesic", "chordal"],
        choices=["geodesic", "chordal"],
        help="Grassmann metrics to report; the second is a cross-check on the first.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--save-dir", default="results")
    p.add_argument("--out", default=f"results_{date.today():%Y_%m_%d}.txt")
    args = p.parse_args()

    conditions = {}
    for token in args.condition:
        group = parse_group(token)
        conditions[group.label] = group
    contrasts = [parse_contrast(t, conditions) for t in args.contrast]
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    print(
        f"{args.effector}: k={args.k}, bin_size={args.bin_size}, "
        f"{len(seeds)} requested seeds, conditions {sorted(conditions)}"
    )
    bases = {}
    for label in sorted({c for contrast in contrasts for c in contrast}):
        bases[label], covered, missing = subspace_bases(
            conditions[label], args.effector, seeds, args.k, args.bin_size
        )
        note = f", missing {missing}" if missing else ""
        print(
            f"  {label}: {len(bases[label])} seeds loaded, "
            f"k={args.k} covers {np.mean(covered):.4f} of trajectory variance{note}"
        )

    records = []
    for metric in args.metrics:
        # The anchor depends only on ambient dimension, k, and metric, so one draw per
        # metric serves every contrast.
        any_basis = next(iter(next(iter(bases.values())).values()))
        null_mean, null_sd = analysis.random_subspace_distance(
            any_basis.shape[0], args.k, metric=metric
        )
        null_angle = typical_angle(null_mean, args.k, metric)
        suffix = f" ({null_angle:.1f} deg)" if null_angle is not None else ""
        print(
            f"\n-- {metric} --  random-subspace null {null_mean:.3f} +/- {null_sd:.3f}{suffix}"
        )

        for contrast in contrasts:
            record = run_contrast(
                contrast, bases, seeds, args.k, metric, args.save_dir, args.effector
            )
            if record is None:
                continue
            parts = []
            for label in contrast:
                angle = typical_angle(record["within"][label], args.k, metric)
                shown = f" ({angle:.1f} deg)" if angle is not None else ""
                parts.append(f"within-{label} {record['within'][label]:.3f}{shown}")
            print(
                f"  {contrast[0]}-vs-{contrast[1]}: "
                + "  ".join(parts)
                + f"  between {record['between']:.3f}"
                + f"  | observed {record['observed']:+.3f}, p={record['p_value']:.2e}"
                + f" ({record['n_pairs']} pairs)"
            )
            records.append(record)

    if not records:
        print("\nNo contrast produced a result; nothing to correct or write.")
        return

    table = format_table(records, args.alpha)
    print("\n" + table)
    with open(args.out, "a") as fh:
        fh.write(f"\n{'=' * 90}\n")
        fh.write("Learning-subspace (Grassmann) geometry, paired permutation\n")
        fh.write("positive observed => second condition clusters tighter than first\n")
        fh.write(
            f"date {date.today():%Y-%m-%d}  effector {args.effector}  k={args.k}  "
            f"bin_size={args.bin_size}  alpha {args.alpha}  "
            f"Holm-Bonferroni over m={len(records)} tests\n"
        )
        fh.write("=" * 90 + "\n")
        fh.write(table + "\n")
    print(f"\nAppended to {args.out}  (Holm family size m={len(records)})")


if __name__ == "__main__":
    main()
