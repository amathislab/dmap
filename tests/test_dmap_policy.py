import gym
import pytest
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from dmap.models.dmap import DMAPPolicyModel
from collections import namedtuple


ObsSpace = namedtuple("obs_space", "original_space,")
seq_len = 30
state_size = 28
action_size = 8
batch_size = 32

model_config = {
    "custom_model": "attention_policy",
    "fcnet_activation": "relu",
    "feature_convnet_params": [
        {"num_filters": 32, "kernel_size": 5, "stride": 4},
        {"num_filters": 32, "kernel_size": 3, "stride": 1},
        {"num_filters": 32, "kernel_size": 3, "stride": 1},
    ],
    "feature_fcnet_hiddens": [128, 32],
    "policy_hiddens": [32, 32],
    "embedding_size": 4,
}

action_space = gym.spaces.box.Box(low=-1, high=1, shape=(action_size,))
x_prev_space = gym.spaces.box.Box(
    low=-float("inf"), high=float("inf"), shape=(seq_len * state_size,)
)
x_t_space = gym.spaces.box.Box(
    low=-float("inf"), high=float("inf"), shape=(state_size,)
)
obs_space = ObsSpace({"x_prev": x_prev_space, "a_t": action_space, "x_t": x_t_space})
random_input_dict = {
    "obs": {
        "x_t": torch.randn(batch_size, state_size),
        "a_t": torch.randn(batch_size, action_size),
        "x_prev": torch.randn(batch_size, seq_len * state_size),
        "a_prev": torch.randn(batch_size, seq_len * action_size),
    }
}

model = DMAPPolicyModel(
    action_space=action_space,
    obs_space=obs_space,
    model_config=model_config,
    num_outputs=action_size,
    name="attention",
)


def test_output_size():
    action_batch, _ = model(random_input_dict)
    assert action_batch.shape == (batch_size, action_size)


def test_input_creation():
    adapt_input, state_input = model.get_adapt_and_state_input(random_input_dict)
    assert adapt_input.shape == ((state_size + action_size) * batch_size, 1, seq_len)
    assert state_input.shape == (batch_size, state_size + action_size)
    assert (
        adapt_input[:state_size, :, :].squeeze()
        == random_input_dict["obs"]["x_prev"][0]
        .reshape(seq_len, state_size)
        .transpose(0, 1)
    ).all()
    assert (
        adapt_input[state_size : state_size + action_size :, :].squeeze()
        == random_input_dict["obs"]["a_prev"][0]
        .reshape(seq_len, action_size)
        .transpose(0, 1)
    ).all()
    assert (
        adapt_input[
            state_size + action_size : 2 * state_size + action_size, :, :
        ].squeeze()
        == random_input_dict["obs"]["x_prev"][1]
        .reshape(seq_len, state_size)
        .transpose(0, 1)
    ).all()
    assert (
        adapt_input[
            2 * state_size + action_size : 2 * state_size + 2 * action_size :, :
        ].squeeze()
        == random_input_dict["obs"]["a_prev"][1]
        .reshape(seq_len, action_size)
        .transpose(0, 1)
    ).all()


def test_keys():
    random_adapt_input = torch.randn(
        (batch_size * (state_size + action_size), 1, seq_len)
    )
    keys, _ = model.get_keys_and_values(random_adapt_input)
    out = torch.sum(keys, dim=2)
    assert out.allclose(torch.ones_like(out))
