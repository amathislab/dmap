import os
import torch
import numpy as np
from definitions import ROOT_DIR
from torch.utils.data import DataLoader
from ray.rllib import rollout
from src.helpers.loading import get_env_and_config, get_trainer
from src.helpers.data_saver import DataSaver
from src.helpers.rma import (
    transfer_policy_weights,
    AdaptModule,
    build_dataset,
    train_loop,
    test_loop,
)


"""
Use the RMA algorithm (https://arxiv.org/abs/2107.04034), adapted to work with SAC,
to train an agent to imitate the Oracle policy. It requires a pretrained Oracle to
work. As an example, we provide the checkpoint of a pretrained Oracle for the 
Half Cheetah environment, sigma = 0.1 and seed = 2.

The code, with the default parameters, collects 20 episodes per training iterations,
which are used to generte a dataset of morphological encodings with the Oracle
policy. The TCN is trained with supervised learning on such dataset for 10 epochs.
The process is repeated for 10 training iterations. The new episodes are added to
the dataset and do not replace the old ones, meaning that the dataset increases in
size with the number of iterations.

The code will look for the saved checkpoint in the folder
"data/{env_name}/pretrained/oracle_sigma_{sigma_literal}_seed_{seed}/checkpoint",
so make sure it is there. Furthermore, it expects the configuration file to be in the folder
"data/{env_name}/pretrained/oracle_sigma_{sigma_literal}_seed_{seed}_data". Please see
the example provided for half_cheetah sigma 0.1 seed 2 in "data/half_cheetah/pretrained.
"""

env_name = "half_cheetah"  # walker, half_cheetah, hopper, ant
sigma = 0.1  # 0.1, 0.3, 0.5
seed = 2  # 0, 1, 2, 3, 4

num_episodes = 20  # 20 - Before resuming training, collect new episodes with the current adapt module
num_training_steps = 10  # 10  # One step is num_epochs_per_training epochs on the currently collected transitions
num_epochs_per_training = 10  # 10
learning_rate = 1e-3
batch_size = 32


device = "cuda" if torch.cuda.is_available() else "cpu"

sigma_literal = str(sigma).replace(".", "")
out_name_specs = f"sigma_{sigma_literal}_seed_{seed}"
checkpoint_path = os.path.join(
    ROOT_DIR,
    "data",
    env_name,
    "pretrained",
    f"oracle_sigma_{sigma_literal}_seed_{seed}",
    "checkpoint",
    "checkpoint",
)

config_folder_path = "_".join((checkpoint_path.split("/checkpoint")[0], "data"))
config_file_name = [
    filename for filename in os.listdir(config_folder_path) if ".json" in filename
][0]

config_path_phase_1 = os.path.join(config_folder_path, config_file_name)

config_path_phase_2 = os.path.join(
    ROOT_DIR,
    "configs",
    env_name,
    f"tcn_{env_name}.json",
)

env_phase_1, config_phase_1 = get_env_and_config(config_path_phase_1)
env_phase_2, config_phase_2 = get_env_and_config(
    config_path_phase_2,
    update_params={
        "env_config": {"sigma": sigma},
        "extra_trainer_params": {"seed": seed},
    },
)
if device == "cuda":
    config_phase_1.num_gpus = 1
    config_phase_2.num_gpus = 1
trainer_phase_1 = get_trainer(config_phase_1, checkpoint_path)
trainer_phase_2 = get_trainer(config_phase_2)

transfer_policy_weights(trainer_phase_1, trainer_phase_2)
checkpoint_phase_2_path = os.path.join(
    ROOT_DIR,
    "data",
    env_name,
    "rma",
    f"{out_name_specs}",
    f"step_0",
)
trainer_phase_2.save(checkpoint_phase_2_path)
config_phase_2.save(os.path.join(checkpoint_phase_2_path, "config.json"))

# Test whether the weight transfer is successful
policy_1 = trainer_phase_1.get_policy()
policy_2 = trainer_phase_2.get_policy()

model_config = policy_1.model.action_model.model_config
encoding_size = model_config["encoding_size"]
action_size = np.prod(env_phase_1.action_space.shape)
obs_size = np.prod(env_phase_1.observation_space.spaces["x_t"].shape)
policy_input = torch.zeros((1, encoding_size + action_size + obs_size), device=device)
features_1 = policy_1.model.action_model._policy_hidden_layers(policy_input)
features_2 = policy_2.model.action_model._policy_hidden_layers(policy_input)

print("Features equal? ", (features_1 == features_2).cpu().numpy().all())

logits_1 = policy_1.model.action_model._policy_logits(features_1)
logits_2 = policy_2.model.action_model._policy_logits(features_2)

print("Logits equal? ", (logits_1 == logits_2).cpu().numpy().all())

old_weights = trainer_phase_2.get_weights()["default_policy"]

import warnings

warnings.filterwarnings("ignore")

saver = DataSaver()  # Here we accumulate the transitions
model_1 = policy_1.model.action_model
model_2 = policy_2.model.action_model
adapt_module = AdaptModule(
    model_2._adapt_fc_layers,
    model_2._adapt_conv_layers,
    model_2._encoder_logits,
)
optimizer = torch.optim.Adam(adapt_module.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

saver_data_len = 0
reward_list = []
loss_list = []
for training_step in range(num_training_steps):
    # First collect new trajectories with the current adaptation module
    rollout.rollout(
        agent=trainer_phase_2,
        env_name=None,
        num_steps=None,
        saver=saver,
        num_episodes=num_episodes,
    )
    reward_training_step = np.mean([x.reward for x in saver.data[saver_data_len:]])
    print(f"Average reward training step {training_step}: {reward_training_step}")
    reward_list.append(reward_training_step)
    saver_data_len = len(saver.data)
    dataset = build_dataset(saver, model_1, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for t in range(num_epochs_per_training):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, adapt_module, loss_fn, optimizer, verbose=True)
        loss_full_dataset = test_loop(dataloader, adapt_module, loss_fn)
    loss_list.append(loss_full_dataset)
    checkpoint_phase_2_path = os.path.join(
        ROOT_DIR,
        "data",
        env_name,
        "rma",
        f"{out_name_specs}",
        f"step_{training_step + 1}",
    )
    trainer_phase_2.save(checkpoint_phase_2_path)
    config_phase_2.save(os.path.join(checkpoint_phase_2_path, "config.json"))

# Which weights were updated?
print("The following weights were updated:")
new_weights = trainer_phase_2.get_weights()["default_policy"]
for key in new_weights:
    if (new_weights[key] != old_weights[key]).any():
        print(key)
