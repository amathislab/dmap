import os
import pickle
import json
from definitions import ROOT_DIR
from src.envs.environment_factory import EnvironmentFactory
from src.helpers.loading import get_env_and_config, get_trainer
from src.helpers.attention import compute_attention_score

"""Create the attention datasets by running a trained DMAP agent on the test xmls. The
agent is only evaluated in the IID scenario (sigma train = sigma test). The output consists
in the list of the attention matrices K, one for each step of the test episodes.

The output will be saved at "data/{env_name}/attention".

The code looks for the saved checkpoint in the folder
"data/{env_name}/pretrained/dmap_sigma_{sigma_literal}_seed_{seed}/checkpoint",
so make sure it is there. Furthermore, it expects the configuration file to be in the folder
"data/{env_name}/pretrained/dmap_sigma_{sigma_literal}_seed_{seed}_data". Please see
the example provided for ant dmap sigma 0.1 seed 2 in "data/ant/pretrained.
"""

env_name = "ant"  # "ant", "hopper", "walker", "half_cheetah"
sigma = 0.1  # 0.1, 0.3, 0.5
seed = 2  # 0, 1, 2, 3, 4

sigma_literal = str(sigma).replace(".", "")
out_name_specs = f"{env_name}_sigma_{sigma_literal}_seed_{seed}"

xml_folder_path = os.path.join(ROOT_DIR, "data", "xmls", env_name)

folder_names_list = [
    name for name in os.listdir(xml_folder_path) if "test_" in name
]

results_dict = {}
agent = "_".join(
    (
        env_name,
        "attention",
        "sigma",
        sigma_literal,
        "seed",
        str(seed),
    )
)
checkpoint_path = os.path.join(
    ROOT_DIR,
    "data",
    env_name,
    "pretrained",
    f"dmap_sigma_{sigma_literal}_seed_{seed}",
    "checkpoint",
    "checkpoint",
)
config_folder_path = "_".join((checkpoint_path.split("/checkpoint")[0], "data"))
config_file_name = [
    filename for filename in os.listdir(config_folder_path) if ".json" in filename
][0]
config_file_path = os.path.join(config_folder_path, config_file_name)

env, config = get_env_and_config(config_file_path)

print(
    "Creating trainer for agent:",
    agent,
)
trainer = get_trainer(config, checkpoint_path=checkpoint_path)
policy = trainer.get_policy()
results_dict[agent] = {
    "data": {},
    "checkpoint": checkpoint_path,
    "config": config_file_path,
}

for folder_name in folder_names_list:
    if sigma_literal in folder_name:
        folder_path = os.path.join(ROOT_DIR, "data", "xmls", env_name, folder_name)
        results_dict[agent]["data"][folder_path] = []

        # Get the test perturbations from a saved list
        with open(
            os.path.join(folder_path, "perturbation_summary.pkl"), "rb"
        ) as file:
            perturbation_summary = pickle.load(file)
        assert perturbation_summary["perturbations"] == env.perturbation_list

        for perturbation_vals in perturbation_summary["values"]:
            episode_data = []
            print("Testing on perturbations", perturbation_vals)
            env = EnvironmentFactory.create(
                config.env_name,
                sigma=1,  # This way the perturbation values are not rescaled
                perturbation_vals=perturbation_vals,
                render=False,
            )
            obs = env.reset()
            done = False
            while not done:
                action = trainer.compute_single_action(obs, explore=False)
                attention_score = compute_attention_score(
                    policy.model.action_model, obs
                )
                attention_score_numpy = attention_score.numpy().squeeze()
                episode_data.append(
                    {
                        "obs": obs["x_t"].numpy().tolist(),
                        "action": action.tolist(),
                        "attention": attention_score_numpy.tolist(),
                    }
                )
                obs, reward, done, info = env.step(action)
            results_dict[agent]["data"][folder_path].append(episode_data)
out_folder = os.path.join(
    ROOT_DIR,
    "data",
    env_name,
    "attention",
)
os.makedirs(out_folder, exist_ok=True)
out_path = os.path.join(
    out_folder,
    f"attention_{out_name_specs}.json",
)
with open(out_path, "w") as file:
    json.dump(results_dict, file)

print("Results dict file created at ", out_path)
