import os
import shutil
import time
import ray
import gym
from src.helpers.data_saver import DataSaver
from datetime import datetime
from ray.rllib import rollout
from ray.tune import tune
from ray.tune.logger import UnifiedLogger
from definitions import ROOT_DIR
from src.rllib_envs.environment_factory import EnvironmentFactory
from src.helpers.experiment_config import ExperimentConfig
from src.models.model_factory import ModelFactory


config_path = os.path.join(
    ROOT_DIR,
    "configs",
    "walker",
    "attention_walker_sigma_03.json",
    # "hopper",
    # "attention_hopper_sigma_03.json",
    # "half_cheetah",
    # "sigma_random_perturbation_half_cheetah_sigma_05.json",
)
config = ExperimentConfig(config_path)
env = EnvironmentFactory.register(config.env_name, **config.env_config)
for policy in config.policy_configs.values():
    for model_params in policy.values():
        if isinstance(model_params, dict):
            model_name = model_params.get("custom_model")
            if model_name is not None:
                ModelFactory.register(model_name)

assert len(config.policy_configs) == 1
policy_id, policy_config = next(iter(config.policy_configs.items()))

trainer_config = config.get_trainer_config_single_agent()

ray.init()

if config.train:
    print("Train mode: trying to learn the policy")
    run_name = os.path.join("training", datetime.today().strftime("%Y-%m-%d"))
    run_dir = os.path.join(config.logdir, run_name)
    trial_dirname = "_".join(
        (config.trial_name, datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    experiment_logdir = os.path.join(run_dir, trial_dirname)
    out_path = "_".join((experiment_logdir, "data"))
    os.makedirs(out_path, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), out_path)
    shutil.copy(config_path, out_path)
    res = tune.run(
        config.trainer_cls,
        name=run_name,
        stop={"episodes_total": config.episodes_total},
        config=trainer_config,
        local_dir=config.logdir,
        trial_dirname_creator=lambda _: trial_dirname,
        checkpoint_freq=config.checkpoint_freq,
        checkpoint_at_end=True,
        keep_checkpoints_num=100,
        restore=config.restore_checkpoint_path,
    )
    assert len(res.trials) == 1
    trial_log_dir = res.trials[0].logdir
    time.sleep(1)
    config.save(os.path.join(trial_log_dir, "configs.json"))

else:
    run_name = "testing"
    run_dir = os.path.join(config.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    experiment_logdir = os.path.join(
        run_dir,
        "_".join((config.trial_name, datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))),
    )
    trainer_config["explore"] = False
    trainer_config["num_workers"] = 0
    trainer = config.trainer_cls(
        config=trainer_config,
        logger_creator=lambda config: UnifiedLogger(config, experiment_logdir),
    )
    if config.restore_checkpoint_path is not None:
        trainer.restore(config.restore_checkpoint_path)

    # print(trainer.workers.local_worker())
    # print(trainer.workers.local_worker().env)
    # env = trainer.workers.local_worker().env
    # env.reset()
    # import pybullet as p

    # video_path = os.path.join(ROOT_DIR, "output", "videos", "ant")
    # env.robot._p.startStateLogging(
    #     p.STATE_LOGGING_VIDEO_MP4,
    #     os.path.join(
    #         video_path, "ant_unperturbed_ckpt_1200_" + str(int(time.time())) + ".mp4"
    #     ),
    # )

    saver = DataSaver()

    rollout.rollout(
        agent=trainer,
        env_name=None,
        num_steps=None,
        num_episodes=config.rollout_num_episodes,
        no_render=False,
        saver=saver,
        # video_dir="output/videos/ant"
    )
    # saver.save_df(
    #     os.path.join(ROOT_DIR, "data", "ant", "no_env_model", "1000_episodes_xml_9.csv")
    # )
    config.save(os.path.join(trainer.logdir, "configs.json"))
