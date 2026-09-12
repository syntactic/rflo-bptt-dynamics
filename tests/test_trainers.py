import motornet as mn
import numpy as np
import pytest
import torch
from motornet.environment import CenterOutReach

from rnn import LeakyRNN
from trainers import BPTTTrainer, RFLOTrainer, position_loss

# --- fixtures --------------------------------------------------------------


@pytest.fixture
def make_env():
    """Factory fixture: call this to get a fresh, independent CenterOutReach
    instance. Must be a factory (not a single shared instance) because
    test_seeded_envs_produce_matching_direction_sequences needs two separate
    envs to compare against each other.
    """

    def _make():
        return CenterOutReach(
            effector=mn.effector.ReluPointMass24(), reaching_distance=0.5
        )

    return _make


@pytest.fixture(params=[BPTTTrainer, RFLOTrainer])
def trainer(request, make_env):
    """Builds one trainer, parametrized over both trainer classes.

    The properties under test here (eval_env separation, train() running
    end-to-end) belong to BaseTrainer, so they should hold for both
    subclasses without writing the checks twice.
    """
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
