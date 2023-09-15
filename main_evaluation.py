import argparse
import os
import pickle
import json
from definitions import ROOT_DIR
from pybullet_m.envs.environment_factory import EnvironmentFactory
from dmap.helpers.loading import get_env_and_config, get_trainer
from dmap.helpers.attention import compute_action_without_encoding

"""
Test a pretrained agent on a set of 100 body configurations per perturbation level. Independently
of the training sigma, the agent will be evaluated on the test sets for all the levels
(0.1, 0.3, 0.5, 0.7). This means that the tests will be both IID and OOD, depending on the training
sigma.

The output will be saved at "data/{env_name}/performance", unless the parameter "out_folder" is
modified.

The code looks for the saved checkpoint in the folder
"data/{env_name}/pretrained/{algorithm}_sigma_{sigma_literal}_seed_{seed}/checkpoint",
so make sure it is there. Furthermore, it expects the configuration file to be in the folder
"data/{env_name}/pretrained/{algorithm}_sigma_{sigma_literal}_seed_{seed}_data". Please see
the example provided for half_cheetah oracle sigma 0.1 seed 2 in "data/half_cheetah/pretrained.
"""

env_name = "hopper"  # walker, half_cheetah, hopper, ant
seed = 2 if env_name == 'ant' else (0 if env_name == 'walker' else 1)  # 0, 1, 2, 3, 4
sigma = 0.1  # 0.1, 0.3, 0.5
algorithm = "dmap-icl"  # "simple", "oracle", "rma", "tcn", "dmap", "dmap-ne", "dmap-icl"

STEPS_BF_FREEZING = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    STEPS_BF_FREEZING = args.steps

sigma_literal = str(sigma).replace(".", "")
out_name_specs = f"{algorithm}_seed_{seed}_sigma_{sigma_literal}"
out_folder = os.path.join(ROOT_DIR, "data", env_name, "performance")

xml_folder_path = os.path.join(ROOT_DIR, "data", "xmls", env_name)
folder_names_list = [name for name in os.listdir(xml_folder_path) if "test_" in name]

results_dict = {}
agent = "_".join(
    (
        env_name,
        algorithm,
        "sigma",
        sigma_literal,
        "seed",
        str(seed),
    )
)
if algorithm == "rma":
    config_folder_path = os.path.join(
        ROOT_DIR,
        "data",
        env_name,
        "rma",
        f"sigma_{sigma_literal}_seed_{seed}",
        "step_10",
    )
    checkpoint_path = os.path.join(
        config_folder_path,
        "checkpoint_000000",
        "checkpoint-0",
    )
else:
    if algorithm == "dmap-ne":
        load_algo = "dmap"
    else:
        load_algo = algorithm
    checkpoint_path = os.path.join(
        ROOT_DIR,
        "data",
        env_name,
        "pretrained",
        f"{load_algo}_sigma_{sigma_literal}_seed_{seed}",
        "checkpoint",
        "checkpoint",
    )
    config_folder_path = "_".join((checkpoint_path.split("/checkpoint")[0], "data"))

config_file_name = [
    filename for filename in os.listdir(config_folder_path) if ".json" in filename
][0]
config_file_path = os.path.join(config_folder_path, config_file_name)

if algorithm == "dmap-icl":
    # Transmit steps parameter to model via json config file
    # This is not a pretty way to do it, but it works...
    with open(config_file_path, 'r+') as json_file:
        config_json = json.load(json_file)
        config_json['policy_configs']['policy']['policy_model']['custom_model_config'] = {}
        config_json['policy_configs']['policy']['policy_model']['custom_model_config']['steps'] = STEPS_BF_FREEZING
        json_file.seek(0)
        json.dump(config_json, json_file)
        json_file.truncate()

env, config = get_env_and_config(config_file_path)

print(
    "Creating trainer for agent:",
    agent,
)
trainer = get_trainer(config, checkpoint_path=checkpoint_path)
if algorithm == "dmap-ne":
    policy = trainer.get_policy()
    action_dim = env.env.action_dim
    embedding_dim = policy.model.action_model.embedding_size
    device = "cpu" if config.num_gpus == 0 else "cuda"
    policy_net_list = []
    for i in range(action_dim):
        policy_net = getattr(policy.model.action_model, f"_policy_fcnet_{i}")
        policy_net_list.append(policy_net)

results_dict[agent] = {
    "results": {},
}

for folder_name in folder_names_list:
    folder_path = os.path.join(ROOT_DIR, "data", "xmls", env_name, folder_name)
    results_dict[agent]["results"][folder_name] = []

    # Get the test perturbations from a saved list
    with open(os.path.join(folder_path, "perturbation_summary.pkl"), "rb") as file:
        perturbation_summary = pickle.load(file)
    assert perturbation_summary["perturbations"] == env.perturbation_list

    for perturbation_vals in perturbation_summary["values"]:
        print("Testing on perturbations", perturbation_vals)
        env = EnvironmentFactory.create(
            config.env_name,
            sigma=1,  # This way the perturbation values are not rescaled
            perturbation_vals=perturbation_vals,
            render=False,
        )
        obs = env.reset()
        done = False
        cum_reward = 0
        while not done:
            if algorithm == "dmap-ne":
                action = compute_action_without_encoding(
                    policy_net_list, obs, action_dim, embedding_dim, device
                )
            else:
                action = trainer.compute_single_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            cum_reward += reward
        env.close() # Added to resolve GUI crash when resetting
        results_dict[agent]["results"][folder_name].append(cum_reward)
        print(
            "agent: ",
            agent,
            ", folder_path",
            folder_path,
            ", episode reward: ",
            cum_reward,
        )
os.makedirs(out_folder, exist_ok=True)
out_path = os.path.join(
    out_folder,
    f"results_{out_name_specs}_steps_{STEPS_BF_FREEZING}.json",
)
with open(out_path, "w") as file:
    json.dump(results_dict, file)

print("Results dict file created at ", out_path)