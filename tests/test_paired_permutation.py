"""Tests for the condition-grouped paired-permutation applier (run_paired_permutation).

The statistic itself lives in analysis.paired_permutation_test (covered in
test_analysis); these cover the applier's own logic: parsing a two-group cell,
resolving matrix row order from a sidecar, excluding a seed, and refusing to pair
a lopsided seed set.
"""

import argparse
import json

import numpy as np
import pytest

import run_paired_permutation as rp


def _distance_matrix(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def test_parse_group_suffix_defaults_to_label():
    g = rp.parse_group("varyB=results/evaryB_varyB")
    assert (g.label, g.directory, g.suffix) == ("varyB", "results/evaryB_varyB", "varyB")


def test_parse_group_explicit_suffix():
    g = rp.parse_group("fixB=results/evaryB_fixB=RFLO")
    assert (g.label, g.directory, g.suffix) == ("fixB", "results/evaryB_fixB", "RFLO")


def test_parse_group_rejects_hyphen_in_label():
    # labels are split on '-' to recover the seed, so a hyphen in the label breaks pairing
    with pytest.raises(argparse.ArgumentTypeError):
        rp.parse_group("vary-B=results/evaryB_varyB")


def test_build_cell_rejects_identical_group_labels():
    with pytest.raises(argparse.ArgumentTypeError):
        rp.build_cell(["RigidTendonArm26", "m_matrix.npy", "RFLO=a", "RFLO=b"])


def test_resolve_labels_prefers_sidecar(tmp_path):
    matrix = tmp_path / "eff_condition_dsa_matrix.npy"
    np.save(matrix, np.zeros((4, 4)))
    sidecar = tmp_path / "eff_condition_dsa_labels.json"
    order = ["fixB-0", "fixB-3", "varyB-0", "varyB-3"]
    sidecar.write_text(json.dumps(order))
    cell = rp.Cell("eff", str(matrix), (rp.parse_group("varyB=a"), rp.parse_group("fixB=b")))
    labels, source = rp.resolve_labels(cell, seeds=[0, 1, 2])
    assert source == "sidecar" and labels == order


def test_resolve_labels_constructed_fallback_is_group_major(tmp_path):
    matrix = tmp_path / "eff_activation_dsa_matrix.npy"  # no sidecar next to it
    cell = rp.Cell("eff", str(matrix), (rp.parse_group("BPTT=a"), rp.parse_group("RFLO=a")))
    labels, source = rp.resolve_labels(cell, seeds=[0, 1])
    assert source == "constructed"
    assert labels == ["BPTT-0", "BPTT-1", "RFLO-0", "RFLO-1"]


def test_drop_excluded_removes_only_named_seeds():
    labels = ["fixB-0", "fixB-3", "varyB-0", "varyB-3"]
    keep, kept = rp._drop_excluded(labels, {0})
    assert keep == [1, 3] and kept == ["fixB-3", "varyB-3"]


def test_check_pairing_rejects_lopsided_seed():
    # seed 3 present in fixB but not varyB -> cannot form a pair
    with pytest.raises(ValueError):
        rp._check_pairing(["fixB-0", "fixB-3", "varyB-0"], ["fixB", "varyB"])


def test_dsa_result_reproduces_orientation_and_exclusion(tmp_path):
    # Plant a tight fixB cluster inside a loose varyB cloud, with a stray seed to drop.
    rng = np.random.default_rng(0)
    seeds = list(range(7))  # after dropping one, 6 pairs -> enumeration floor 2/65 < 0.05
    varyB = rng.normal(0.0, 1.0, size=(len(seeds), 2))
    fixB = rng.normal(0.0, 0.01, size=(len(seeds), 2))
    # sidecar/matrix row order: all varyB, then all fixB
    order = [f"varyB-{s}" for s in seeds] + [f"fixB-{s}" for s in seeds]
    points = np.vstack([varyB, fixB])
    matrix = tmp_path / "eff_condition_dsa_matrix.npy"
    np.save(matrix, _distance_matrix(points))
    (tmp_path / "eff_condition_dsa_labels.json").write_text(json.dumps(order))

    # groups (baseline=varyB, hypothesized-tighter=fixB): positive => fixB tighter
    cell = rp.Cell("eff", str(matrix), (rp.parse_group("varyB=a"), rp.parse_group("fixB=b")))
    res = rp.dsa_result(cell, seeds=seeds, exclude={0})
    assert res["label_source"] == "sidecar"
    assert res["n_pairs"] == 6  # seed 0 dropped
    assert res["n_permutations"] == 2**6
    assert res["observed"] > 0  # fixB (second group) clusters tighter
    assert res["p_value"] < 0.05
