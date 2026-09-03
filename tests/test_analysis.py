import numpy as np
import pytest
import torch
from pathlib import Path

from analysis import (
    _to_numpy,
    load_experiment_run,
    load_all_runs,
    extract_per_direction_trajectories,
    process_activation_trajectories,
    process_weight_trajectories,
    summarize_within_between,
    calculate_p_value_of_dsa_distance,
    reach_metrics,
    perp_dist,
    reach_distance,
    terminal_error,
    reach_relative_terminal_error,
)


@pytest.fixture
def mock_run_data():
    """Create a self-consistent mock run artifact dictionary."""
    T = 100
    n_targets = 8
    n_rec = 64
    H = torch.randn(T, n_targets, n_rec)
    direction_idx = torch.arange(n_targets)
    weight_history = [torch.randn(n_rec * n_rec) for _ in range(50)]
    losses = list(np.linspace(1.0, 0.2, 50))
    metrics = [(0, {i: (0.0, 0.1) for i in range(n_targets)}), (50, {i: (1.0, 0.01) for i in range(n_targets)})]
    FT = torch.zeros(T, n_targets, 2)
    Y = torch.zeros(T, n_targets, 6)
    goal = torch.randn(n_targets, 2)

    return {
        "weight_history": weight_history,
        "losses": losses,
        "metrics": metrics,
        "H": H,
        "Y": Y,
        "FT": FT,
        "goal": goal,
        "direction_idx": direction_idx,
        "seed": 0,
        "n_rec": n_rec,
        "num_steps": 50,
        "lr": 0.05,
        "batch_size": 32,
        "effector": "ReluPointMass24",
    }


def test_extract_per_direction_trajectories(mock_run_data):
    trials = extract_per_direction_trajectories(mock_run_data)
    assert len(trials) == 8
    assert trials[0].shape == (100, 64)


def test_extract_per_direction_trajectories_order_assertion(mock_run_data):
    mock_run_data["direction_idx"] = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7])
    with pytest.raises(AssertionError):
        extract_per_direction_trajectories(mock_run_data)


def test_summarize_within_between():
    similarities = np.array([
        [0.0, 0.1, 0.5, 0.6],
        [0.1, 0.0, 0.4, 0.5],
        [0.5, 0.4, 0.0, 0.2],
        [0.6, 0.5, 0.2, 0.0],
    ])
    labels = ["BPTT-0", "BPTT-1", "RFLO-0", "RFLO-1"]
    summary = summarize_within_between(similarities, labels)

    assert ("BPTT", "BPTT") in summary
    assert ("RFLO", "RFLO") in summary
    assert ("BPTT", "RFLO") in summary
    assert np.isclose(summary[("BPTT", "BPTT")], 0.1)
    assert np.isclose(summary[("RFLO", "RFLO")], 0.2)
    assert np.isclose(summary[("BPTT", "RFLO")], 0.5)


def test_calculate_p_value_of_dsa_distance():
    similarities = np.array([
        [0.0, 0.1, 0.9, 0.9],
        [0.1, 0.0, 0.9, 0.9],
        [0.9, 0.9, 0.0, 0.1],
        [0.9, 0.9, 0.1, 0.0],
    ])
    labels = ["BPTT-0", "BPTT-1", "RFLO-0", "RFLO-1"]
    p_val = calculate_p_value_of_dsa_distance(similarities, labels, ("BPTT", "RFLO"))
    # For N=4 (C(4,2)=6 splits), the minimum p-value with symmetric grouping is (2+1)/(6+1) = 3/7
    assert np.isclose(p_val, 3 / 7)


def test_process_weight_trajectories(mock_run_data):
    runs = {
        ("BPTT", 0): mock_run_data,
        ("RFLO", 0): mock_run_data,
    }
    pc_trajectories, pca = process_weight_trajectories(runs, bin_size=5, n_components=0.95)
    assert ("BPTT", 0) in pc_trajectories
    assert ("RFLO", 0) in pc_trajectories
    assert pc_trajectories[("BPTT", 0)].shape[0] == 10  # 50 steps // 5


def test_process_activation_trajectories(mock_run_data):
    runs = {
        ("BPTT", 0): mock_run_data,
        ("RFLO", 0): mock_run_data,
    }
    pc_trajectories, pca = process_activation_trajectories(runs, n_components=0.95)
    assert ("BPTT", 0) in pc_trajectories
    assert len(pc_trajectories[("BPTT", 0)]) == 8
    assert pc_trajectories[("BPTT", 0)][0].shape[0] == 100


def test_perp_dist():
    traj = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    dists = perp_dist(traj, a, b)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 0.5)
    assert np.isclose(dists[2], 0.0)


def test_reach_metrics():
    FT = np.zeros((1, 10, 2))
    FT[0, -1] = np.array([0.1, 0.0])
    targets = np.array([[0.1, 0.0]])
    succ, dev = reach_metrics(FT, targets, success_radius=0.05)
    assert succ == 1.0
    assert np.isclose(dev, 0.0)


def test_reach_distance():
    # Mirrors run_experiment.py: 0.1 for any "Arm" effector, 0.5 otherwise.
    assert reach_distance("ReluPointMass24") == 0.5
    assert reach_distance("RigidTendonArm26") == 0.1


def test_terminal_error():
    # Two targets; final fingertip lands 0.03 short of the first, exactly on the second.
    FT = np.zeros((2, 10, 2))
    FT[0, -1] = np.array([0.07, 0.0])
    FT[1, -1] = np.array([0.0, 0.2])
    targets = np.array([[0.10, 0.0], [0.0, 0.2]])
    err = terminal_error(FT, targets)
    assert err.shape == (2,)
    assert np.isclose(err[0], 0.03)
    assert np.isclose(err[1], 0.0)


def test_reach_relative_terminal_error():
    # Same 0.03 absolute error becomes a different fraction per effector's reach distance.
    FT = np.zeros((1, 10, 2))
    FT[0, -1] = np.array([0.07, 0.0])
    targets = np.array([[0.10, 0.0]])
    rel_pm = reach_relative_terminal_error(FT, targets, "ReluPointMass24")
    rel_arm = reach_relative_terminal_error(FT, targets, "RigidTendonArm26")
    assert np.isclose(rel_pm[0], 0.03 / 0.5)  # 6% of point-mass reach
    assert np.isclose(rel_arm[0], 0.03 / 0.1)  # 30% of arm reach
