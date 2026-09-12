"""Apply the paired (block) permutation test for within-group dispersion
asymmetry to trained runs, across a set of two-group contrasts and both geometry
views (activation DSA distances and weight ΔW cosines), then correct the family
of p-values for multiple comparisons with Holm-Bonferroni.

A contrast is a *cell*: an effector, a saved distance/similarity matrix, and two
groups. Each group is `label=directory[=suffix]`: the label it goes by in the
matrix, the directory its `.pt` artifacts live in, and the filename rule suffix
(defaulting to the label). This one shape covers both contrasts the project runs:

  learning rule    one directory, two suffixes:
    BPTT=results/clean  RFLO=results/clean
  feedback condition (E-varyB)   two directories, one suffix (RFLO):
    varyB=results/evaryB_varyB=RFLO  fixB=results/evaryB_fixB=RFLO

Group order is (baseline, hypothesized-tighter): a positive observed statistic
means the *second* group clusters tighter than the first. For E-varyB the
canalization prediction (shared B tightens the cluster) is fixB tighter, so pass
`varyB=... fixB=...`.

The exchangeable unit is the seed: the two group members for a seed share their
initialization and target stream, so the null flips the group label within each
seed rather than shuffling labels across seeds. See analysis.paired_permutation_test.

Example:
    python run_paired_permutation.py \
        --cell RigidTendonArm26 results/clean/RigidTendonArm26_activation_dsa_matrix.npy \
               BPTT=results/clean RFLO=results/clean \
        --cell RigidTendonArm26 results/RigidTendonArm26_condition_dsa_matrix.npy \
               varyB=results/evaryB_varyB=RFLO fixB=results/evaryB_fixB=RFLO \
        --exclude-seeds 0 --n-seeds 15

Activation DSA is read from the matrix named on the cell; its row order comes from a
`<...>_labels.json` sidecar next to the matrix (written by run_condition_dsa.py),
falling back to a group-major construction from `--seeds` when no sidecar exists. A
cell whose matrix is missing has its DSA view skipped. Weight ΔW cosines are built
here by loading each artifact once and keeping only its flattened ΔW, so the full
weight histories never sit in memory together; a cell whose artifacts carry no
weight history (behavior-only runs) has its ΔW view skipped.
"""

import argparse
import gc
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch

import analysis


@dataclass(frozen=True)
class Group:
    """One side of a contrast: its matrix label, artifact directory, file suffix."""

    label: str
    directory: str
    suffix: str


@dataclass(frozen=True)
class Cell:
    """A two-group contrast on one effector, backed by one saved matrix."""

    effector: str
    matrix_path: str
    groups: tuple  # (Group, Group): (baseline, hypothesized-tighter)


def parse_group(token):
    """Parse a `label=directory[=suffix]` group token; suffix defaults to label.

    The label is used both as the matrix prefix and (via analysis._seed_to_pair,
    which splits labels on '-') as a parse key, so it must not contain '-'.
    """
    parts = token.split("=")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"group '{token}' must be 'label=directory' or 'label=directory=suffix'"
        )
    label, directory = parts[0], parts[1]
    suffix = parts[2] if len(parts) == 3 else label
    if "-" in label:
        raise argparse.ArgumentTypeError(
            f"group label '{label}' must not contain '-' (it is split on '-' to "
            f"recover the seed); use e.g. 'varyB', not 'vary-B'"
        )
    return Group(label=label, directory=directory, suffix=suffix)


def build_cell(tokens):
    """Turn the four --cell nargs tokens into a Cell (validated)."""
    effector, matrix_path, group_a, group_b = tokens
    g0, g1 = parse_group(group_a), parse_group(group_b)
    if g0.label == g1.label:
        raise argparse.ArgumentTypeError(
            f"the two groups of a cell must have distinct labels (got '{g0.label}' twice)"
        )
    return Cell(effector=effector, matrix_path=matrix_path, groups=(g0, g1))


def _sidecar_path(matrix_path):
    """The `<...>_labels.json` companion for a `<...>_matrix.npy` matrix path."""
    p = Path(matrix_path)
    if not p.name.endswith("_matrix.npy"):
        raise ValueError(f"matrix path '{matrix_path}' does not end in '_matrix.npy'")
    return p.with_name(p.name[: -len("_matrix.npy")] + "_labels.json")


def resolve_labels(cell, seeds):
    """Row-order labels for this cell's matrix: the sidecar if present (authoritative
    row order), else built group-major from `seeds` (all of group 0, then group 1)."""
    sidecar = _sidecar_path(cell.matrix_path)
    if sidecar.exists():
        return list(json.loads(sidecar.read_text())), "sidecar"
    return [f"{g.label}-{s}" for g in cell.groups for s in seeds], "constructed"


def _drop_excluded(labels, exclude):
    """Keep only labels whose seed is not in `exclude`; return (kept_indices, kept_labels)."""
    keep = [i for i, lab in enumerate(labels) if int(lab.split("-")[1]) not in exclude]
    return keep, [labels[i] for i in keep]


def _check_pairing(labels, group_labels):
    """Every label prefix is one of the two groups, and every seed carries both,
    otherwise _paired_stat's unconditional pair[group] lookups fail cryptically."""
    prefixes = {lab.split("-")[0] for lab in labels}
    if prefixes != set(group_labels):
        raise ValueError(
            f"label prefixes {sorted(prefixes)} do not match cell groups {sorted(group_labels)}"
        )
    present = defaultdict(set)
    for lab in labels:
        pre, seed = lab.split("-")
        present[seed].add(pre)
    lopsided = {s: sorted(v) for s, v in present.items() if v != set(group_labels)}
    if lopsided:
        raise ValueError(f"seeds missing one group (cannot pair): {lopsided}")


def dsa_result(cell, seeds, exclude):
    """Paired test on the saved activation DSA distance matrix, or None if the
    matrix has not been computed for this cell yet."""
    matrix_path = Path(cell.matrix_path)
    if not matrix_path.exists():
        print(f"  [skip DSA] no matrix at {matrix_path}")
        return None
    M = np.load(matrix_path)
    labels, source = resolve_labels(cell, seeds)
    if M.shape[0] != len(labels):
        raise ValueError(
            f"{matrix_path} is {M.shape[0]}x{M.shape[0]} but {len(labels)} labels "
            f"(source: {source}); row order and seed set must match the saved matrix."
        )
    keep, kept_labels = _drop_excluded(labels, exclude)
    M = M[np.ix_(keep, keep)]
    group_labels = [g.label for g in cell.groups]
    _check_pairing(kept_labels, group_labels)
    res = analysis.paired_permutation_test(
        M, kept_labels, metric_is_similarity=False, rule_kinds=tuple(group_labels)
    )
    res["n_pairs"] = len(kept_labels) // 2
    res["label_source"] = source
    return res


def delta_w_result(cell, seeds, exclude):
    """Paired test on the ΔW cosine matrix, built by loading each artifact once and
    retaining only its unit ΔW vector so the weight histories are freed before the
    next load. None if any artifact carries no weight history (behavior-only run)."""
    labels, source = resolve_labels(cell, seeds)
    _, kept_labels = _drop_excluded(labels, exclude)
    units, dlabels = [], []
    for g in cell.groups:
        for lab in kept_labels:
            pre, seed = lab.split("-")
            if pre != g.label:
                continue
            path = Path(g.directory) / f"{cell.effector}_seed{seed}_{g.suffix}.pt"
            if not path.exists():
                print(f"  [skip dW] missing artifact {path}")
                return None
            run = torch.load(path, map_location="cpu", weights_only=False)
            wh = run.get("weight_history")
            if wh is None or len(wh) < 2:
                n = 0 if wh is None else len(wh)
                print(
                    f"  [skip dW] {path} has weight_history len {n} < 2 (needs a weight-history run)"
                )
                del run
                gc.collect()
                return None
            d = analysis.delta_w(run)
            units.append(d / np.linalg.norm(d))
            dlabels.append(lab)
            del run
            gc.collect()
    U = np.array(units)
    group_labels = [g.label for g in cell.groups]
    _check_pairing(dlabels, group_labels)
    res = analysis.paired_permutation_test(
        U @ U.T, dlabels, metric_is_similarity=True, rule_kinds=tuple(group_labels)
    )
    res["n_pairs"] = len(dlabels) // 2
    res["label_source"] = source
    return res


def format_table(records, alpha):
    """Render the collected results with a Holm-corrected p-value column. Records is
    a list of dicts with keys: cell, metric, observed, p_value, n_perms, n_pairs."""
    reject, p_holm = analysis.holm_bonferroni(
        [r["p_value"] for r in records], alpha=alpha
    )
    header = (
        f"{'cell':<40}{'metric':<16}{'pairs':>6}{'observed':>11}"
        f"{'raw p':>12}{'Holm p':>12}{'reject':>9}{'n_perms':>10}"
    )
    lines = [header, "-" * len(header)]
    for r, ph, rej in zip(records, p_holm, reject):
        lines.append(
            f"{r['cell']:<40}{r['metric']:<16}{r['n_pairs']:>6}{r['observed']:>11.4f}"
            f"{r['p_value']:>12.2e}{ph:>12.2e}{('yes' if rej else 'no'):>9}"
            f"{r['n_perms']:>10}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cell",
        action="append",
        nargs=4,
        metavar=("EFFECTOR", "MATRIX", "GROUP_A", "GROUP_B"),
        required=True,
        help="a contrast: effector, its matrix, then two 'label=dir[=suffix]' groups "
        "(baseline first, hypothesized-tighter second). Repeatable.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=15)
    p.add_argument(
        "--exclude-seeds",
        nargs="+",
        type=int,
        default=[],
        help="seeds to drop from every cell (e.g. a degenerate condition pair). "
        "Applies to all cells in this invocation.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--out",
        default=f"results_{date.today():%Y_%m_%d}.txt",
        help="file to append the results table to",
    )
    args = p.parse_args()

    cells = [build_cell(tokens) for tokens in args.cell]
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    exclude = set(args.exclude_seeds)

    records, skipped = [], []
    for cell in cells:
        cid = f"{cell.effector}:{cell.groups[0].label}-vs-{cell.groups[1].label}"
        print(f"[{cid}]  seeds={seeds}  exclude={sorted(exclude) or 'none'}")
        for metric_name, fn in (
            ("activation_dsa", dsa_result),
            ("delta_w_cosine", delta_w_result),
        ):
            res = fn(cell, seeds, exclude)
            if res is None:
                skipped.append((cid, metric_name))
                continue
            records.append(
                {
                    "cell": cid,
                    "metric": metric_name,
                    "observed": res["observed"],
                    "p_value": res["p_value"],
                    "n_perms": res["n_permutations"],
                    "n_pairs": res["n_pairs"],
                }
            )
            print(
                f"  {metric_name}: observed={res['observed']:+.4f} "
                f"p={res['p_value']:.2e}  {res['n_pairs']} pairs, "
                f"{res['n_permutations']} sign-flips ({res['label_source']} labels)"
            )

    if not records:
        print("No cells produced results; nothing to correct or write.")
        return

    m = len(records)
    table = format_table(records, args.alpha)
    print("\n" + table)
    if skipped:
        print(
            "\nSkipped views (not in the Holm family): "
            + ", ".join(f"{c}/{v}" for c, v in skipped)
        )

    with open(args.out, "a") as fh:
        fh.write(f"\n{'=' * 90}\n")
        fh.write("Paired permutation test (within-group dispersion asymmetry)\n")
        fh.write(
            "positive observed => second group (GROUP_B) clusters tighter than first (GROUP_A)\n"
        )
        fh.write(
            f"date {date.today():%Y-%m-%d}  alpha {args.alpha}  "
            f"Holm-Bonferroni over m={m} tests"
        )
        if exclude:
            fh.write(f"  (excluded seeds: {sorted(exclude)})")
        fh.write("\n")
        if skipped:
            fh.write(
                "skipped views: " + ", ".join(f"{c}/{v}" for c, v in skipped) + "\n"
            )
        fh.write("=" * 90 + "\n")
        fh.write(table + "\n")
    print(f"\nAppended to {args.out}  (Holm family size m={m})")


if __name__ == "__main__":
    main()
