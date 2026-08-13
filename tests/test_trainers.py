import numpy as np
import pytest
import torch
import motornet as mn
from motornet.environment import CenterOutReach

from rnn import LeakyRNN
from trainers import BPTTTrainer, RFLOTrainer, position_loss


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def make_env():
    """Factory fixture: call this to get a *fresh*, independent CenterOutReach
    instance. Must be a factory (not a single shared instance) because
    test_seeded_envs_produce_matching_direction_sequences needs two separate
    envs to compare against each other.
    """
    def _make():
        # TODO: construct and return a CenterOutReach with a small/fast
        # effector (e.g. ReluPointMass24), matching the params used in the
        # earlier smoke test (reaching_distance, n_targets, max_ep_duration).
        return CenterOutReach(effector=mn.effector.ReluPointMass24(), reaching_distance=0.5)
    return _make


@pytest.fixture(params=[BPTTTrainer, RFLOTrainer])
def trainer(request, make_env):
    """Builds one trainer, parametrized over both trainer classes -- the
    properties under test here (eval_env separation, train() running
    end-to-end) belong to BaseTrainer, so they should hold for both
    subclasses without writing the checks twice.
    """
    trainer_cls = request.param
    # TODO: build a LeakyRNN sized to match make_env()'s observation/action
    # space, construct trainer_cls(net, env, position_loss, lr=..., device="cpu"),
    # and return it.
    center_out = make_env()
    rnn = LeakyRNN(n_in=center_out.observation_space.shape[0],
                   n_rec=64, n_out=center_out.n_muscles)
    return trainer_cls(rnn, center_out, position_loss)


# --- tests -------------------------------------------------------------------

def test_eval_env_is_distinct_from_training_env(trainer):
    assert trainer.eval_env is not trainer.env
    


def test_eval_env_usable_after_deepcopy(trainer):
    # TODO: call trainer.eval_env.reset(seed=..., options={"batch_size": ...})
    # and check the returned obs has the expected shape (batch dim matches).
    obs, info = trainer.eval_env.reset(seed=0, options={"batch_size":4})
    assert obs.shape == (4, trainer.eval_env.observation_space.shape[0])


def test_train_runs_end_to_end(trainer):
    # TODO: call trainer.train(num_steps=..., batch_size=..., eval_every=...,
    # seed=...) with small numbers (this should be a *fast* smoke test, not a
    # real training run). Assert:
    #   - all losses are finite (np.isfinite)
    #   - len(trainer.weight_history) == num_steps
    losses, _ = trainer.train(num_steps=4, batch_size=4)
    assert all(np.isfinite(losses))
    assert len(trainer.weight_history) == 4


def test_seeded_envs_produce_matching_direction_sequences(make_env):
    # TODO: build two independent envs via make_env(). Seed both once with the
    # *same* seed value (env.reset(seed=42, options={"batch_size": ...})).
    # Then call env.reset(options={"batch_size": ...}) (no seed!) on each
    # several times in a lockstep loop, collecting info["direction_idx"] from
    # each call. Assert the two collected sequences are equal every time
    # (torch.equal per step, or stack + torch.equal on the whole sequence).
    #
    # This is the property train()/train_step's seed-once design depends on:
    # two separately-constructed envs, seeded identically once, must draw the
    # identical sequence of directions afterward with no further seeding.
    co1 = make_env()
    co2 = make_env()
    options = {"batch_size": 16}

    co1.reset(seed=42, options=options)
    co2.reset(seed=42, options=options)

    for i in range(8):
        _, info1 = co1.reset(options=options)
        _, info2 = co2.reset(options=options)

        assert torch.equal(info1["direction_idx"], info2["direction_idx"])
