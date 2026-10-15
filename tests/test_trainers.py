from pathlib import Path

import motornet as mn
import numpy as np
import pytest
import torch
from motornet.environment import CenterOutReach

from rnn import LeakyRNN
from trainers import BPTTTrainer, RFLOTrainer, make_feedback_matrix, position_loss

# --- fixtures --------------------------------------------------------------


@pytest.fixture
def make_env():
    """Returns a factory for fresh CenterOutReach envs, since some tests need two
    independent ones."""

    def _make():
        return CenterOutReach(
            effector=mn.effector.ReluPointMass24(), reaching_distance=0.5
        )

    return _make


@pytest.fixture(params=[BPTTTrainer, RFLOTrainer])
def trainer(request, make_env):
    """One trainer of each class. The behavior tested with it lives in
    BaseTrainer, so it should hold for both."""
    trainer_cls = request.param
    center_out = make_env()
    rnn = LeakyRNN(
        n_in=center_out.observation_space.shape[0], n_rec=64, n_out=center_out.n_muscles
    )
    return trainer_cls(rnn, center_out, position_loss)


# --- tests -------------------------------------------------------------------


def test_eval_env_is_distinct_from_training_env(trainer):
    assert trainer.eval_env is not trainer.env


def test_eval_env_usable_after_deepcopy(trainer):
    obs, info = trainer.eval_env.reset(seed=0, options={"batch_size": 4})
    assert obs.shape == (4, trainer.eval_env.observation_space.shape[0])


def test_train_runs_end_to_end(trainer):
    losses, _ = trainer.train(num_steps=4, batch_size=4)
    assert all(np.isfinite(losses))
    assert len(trainer.weight_history) == 4


def test_seeded_envs_produce_matching_direction_sequences(make_env):
    co1 = make_env()
    co2 = make_env()
    options = {"batch_size": 16}

    co1.reset(seed=42, options=options)
    co2.reset(seed=42, options=options)

    for i in range(8):
        _, info1 = co1.reset(options=options)
        _, info2 = co2.reset(options=options)

        assert torch.equal(info1["direction_idx"], info2["direction_idx"])


# --- gradient projection -----------------------------------------------------

WEIGHT_NAMES = ("W_rec", "W_in", "W_out")


def _make_net(env, seed=0):
    torch.manual_seed(seed)
    return LeakyRNN(n_in=env.observation_space.shape[0], n_rec=64, n_out=env.n_muscles)


def _weights(net):
    return {name: getattr(net, name).detach().clone() for name in WEIGHT_NAMES}


def _projector(B):
    """Projector onto span(B), computed independently of the trainer's."""
    q, _ = torch.linalg.qr(B, mode="reduced")
    return q @ q.T


def _residual_outside_span(dW, B):
    """Fraction of ||dW|| whose columns lie outside span(B)."""
    return (torch.linalg.norm(dW - _projector(B) @ dW) / torch.linalg.norm(dW)).item()


def _train_and_diff(make_env, grad_basis, num_steps, net_seed=0):
    """Train a fresh BPTT net and return the change in each weight matrix.
    Calls with the same net_seed share init and target batches."""
    env = make_env()
    net = _make_net(env, seed=net_seed)
    before = _weights(net)
    # The update is a difference of float32 weights, so its round-off scales
    # with the weights. lr=1.0 keeps it well above that; at lr=0.01 round-off
    # alone gives a relative error of ~1e-3.
    trainer = BPTTTrainer(net, env, position_loss, lr=1.0, grad_basis=grad_basis)
    trainer.train(num_steps=num_steps, batch_size=4, seed=0)
    after = _weights(net)
    return {name: after[name] - before[name] for name in WEIGHT_NAMES}


def _point_mass_B():
    return make_feedback_matrix(n_rec=64, n_out=4, seed=0)


def test_projected_bptt_update_stays_in_span(make_env):
    # Several steps: a sum of updates in span(B) should stay in span(B)
    B = _point_mass_B()
    dW = _train_and_diff(make_env, grad_basis=B, num_steps=5)

    assert _residual_outside_span(dW["W_rec"], B) < 1e-4
    assert _residual_outside_span(dW["W_in"], B) < 1e-4
    # a zero update would pass trivially
    assert torch.linalg.norm(dW["W_rec"]) > 0
    assert torch.linalg.norm(dW["W_in"]) > 0
    assert torch.linalg.norm(dW["W_out"]) > 0


def test_unprojected_bptt_leaves_span(make_env):
    # Negative control. A gradient unrelated to B keeps about n_out / n_rec of
    # its energy in span(B), so the residual should be near sqrt(1 - 4/64) = 0.97.
    B = _point_mass_B()
    dW = _train_and_diff(make_env, grad_basis=None, num_steps=5)

    assert _residual_outside_span(dW["W_rec"], B) > 0.8


def test_projected_step_is_projection_of_plain_step(make_env):
    # Checks the update is the projected gradient step, not just any update in
    # span(B). One step only, since the two nets diverge after it.
    B = _point_mass_B()
    P = _projector(B)
    plain = _train_and_diff(make_env, grad_basis=None, num_steps=1)
    projected = _train_and_diff(make_env, grad_basis=B, num_steps=1)

    for name in ("W_rec", "W_in"):
        expected = P @ plain[name]
        error = torch.linalg.norm(projected[name] - expected)
        assert error / torch.linalg.norm(expected) < 1e-4
    # the readout is left free, as in RFLO
    assert torch.allclose(projected["W_out"], plain["W_out"], rtol=1e-5, atol=1e-9)


def test_feedback_matrix_matches_saved_fixB_run():
    # Projected runs are paired with the fixB RFLO runs on disk, so the
    # regenerated B must match the one they used.
    path = Path("results/evaryB_fixB_wh/RigidTendonArm26_seed1_RFLO.pt")
    if not path.exists():
        pytest.skip(f"{path} not available")
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    B_saved = saved["feedback_matrix"]

    B = make_feedback_matrix(n_rec=B_saved.shape[0], n_out=B_saved.shape[1], seed=0)
    assert torch.equal(B, B_saved)
