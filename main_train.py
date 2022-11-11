import os
import shutil
import ray
from datetime import datetime
from ray.tune import tune
from definitions import ROOT_DIR
from pybullet_m.envs.environment_factory import EnvironmentFactory
from dmap.helpers.experiment_config import ExperimentConfig
from dmap.models.model_factory import ModelFactory

""">
Train an agent in the morphological perturbation environments with SAC. The parameters of the algorithm
and of the network architecture are defined in the configuration json file. 

To change agent and algorithm, simply choose a different configuration file from the folder "configs"
and assign its path to the variable "config_path".

To change other parameters, such as sigma, random seed, network size, ... please change the configuration
file directly.

The training logs information to the folder "output/training", including a checkpoint every 100 training
iterations. Depending on the agent and the algorithm, a single training might take approx. 10-30 hours on
a 10-core cpu. To benefit from gpu acceleration, change the number of available gpus in the config file.
"""
config_path = os.path.join(
    ROOT_DIR,
    "configs",
    "walker",  # "ant", "walker", "hopper", "half_cheetah"
    "simple_walker.json",  # "simple", "oracle", "tcn", "dmap"
)
config = ExperimentConfig(config_path)
env = EnvironmentFactory.register(config.env_name, **config.env_config)
ModelFactory.register_models_from_config(config.policy_configs)

assert (
    len(config.policy_configs) == 1
), f"This script can only run single agent trainings, but {len(config.policy_configs)} policies were given"
policy_id, policy_config = next(iter(config.policy_configs.items()))

trainer_config = config.get_trainer_config()

ray.init()

print("Train mode: optimizing the policy")
run_name = os.path.join("training", datetime.today().strftime("%Y-%m-%d"))
run_dir = os.path.join(ROOT_DIR, config.logdir, run_name)
trial_dirname = "_".join(
    (config.trial_name, datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
)

# Copy the configuration and the main files in the out dir for reproducibility
experiment_logdir = os.path.join(run_dir, trial_dirname)
out_path = "_".join((experiment_logdir, "data"))
os.makedirs(out_path, exist_ok=True)
shutil.copy(os.path.abspath(__file__), out_path)
shutil.copy(config_path, out_path)

# Start the training with ray tune
res = tune.run(
    config.trainer_cls,
    name=run_name,
    stop={"episodes_total": config.episodes_total},
    config=trainer_config,
    local_dir=os.path.join(ROOT_DIR, config.logdir),
    trial_dirname_creator=lambda _: trial_dirname,
    checkpoint_freq=config.checkpoint_freq,
    checkpoint_at_end=True,
    keep_checkpoints_num=100,
    restore=config.restore_checkpoint_path,
)
