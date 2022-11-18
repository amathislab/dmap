import gym
import pybullet_m.envs.ant
import pybullet_m.envs.hopper
import pybullet_m.envs.walker
import pybullet_m.envs.half_cheetah
from ray.tune import register_env


class EnvironmentFactory:
    """Static factory to instantiate and register gym environments by name."""

    @staticmethod
    def create(env_name, **kwargs):
        """Creates an environment given its name as a string, and forwards the kwargs
        to its __init__ function.

        Args:
            env_name (str): name of the environment

        Raises:
            ValueError: if the name of the environment is unknown

        Returns:
            gym.env: the selected environment
        """
        if env_name == "SimpleAntBulletEnv":
            return gym.make("RandomPerturbationAntBulletEnv-v0", **kwargs)
        elif env_name == "SimpleHopperBulletEnv":
            return gym.make("RandomPerturbationHopperBulletEnv-v0", **kwargs)
        elif env_name == "SimpleWalker2DBulletEnv":
            return gym.make("RandomPerturbationWalker2DBulletEnv-v0", **kwargs)
        elif env_name == "SimpleSymmetricWalker2DBulletEnv":
            return gym.make("SymmetricRandomPerturbationWalker2DBulletEnv-v0", **kwargs)
        elif env_name == "SimpleHalfCheetahBulletEnv":
            return gym.make("RandomPerturbationHalfCheetahBulletEnv-v0", **kwargs)
        elif env_name == "OracleAntBulletEnv":
            return gym.make("RMAAntBulletEnv-v0", **kwargs)
        elif env_name == "OracleHopperBulletEnv":
            return gym.make("RMAHopperBulletEnv-v0", **kwargs)
        elif env_name == "OracleWalkerBulletEnv":
            return gym.make("RMAWalker2DBulletEnv-v0", **kwargs)
        elif env_name == "OracleSymmetricWalkerBulletEnv":
            return gym.make("RMASymmetricWalker2DBulletEnv-v0", **kwargs)
        elif env_name == "OracleHalfCheetahBulletEnv":
            return gym.make("RMAHalfCheetahBulletEnv-v0", **kwargs)
        elif env_name == "AdaptAntBulletEnv":
            return gym.make("RMAAntBulletEnv-v0", include_adapt_state=True, **kwargs)
        elif env_name == "AdaptHopperBulletEnv":
            return gym.make("RMAHopperBulletEnv-v0", include_adapt_state=True, **kwargs)
        elif env_name == "AdaptWalkerBulletEnv":
            return gym.make(
                "RMAWalker2DBulletEnv-v0", include_adapt_state=True, **kwargs
            )
        elif env_name == "AdaptSymmetricWalkerBulletEnv":
            return gym.make(
                "RMASymmetricWalker2DBulletEnv-v0", include_adapt_state=True, **kwargs
            )
        elif env_name == "AdaptHalfCheetahBulletEnv":
            return gym.make(
                "RMAHalfCheetahBulletEnv-v0", include_adapt_state=True, **kwargs
            )
        elif env_name == "CustomAntBulletEnv":
            return gym.make("CustomAntBulletEnv-v0", **kwargs)
        elif env_name == "CustomHopperBulletEnv":
            return gym.make("CustomHopperBulletEnv-v0", **kwargs)
        elif env_name == "CustomWalkerBulletEnv":
            return gym.make("CustomWalker2DBulletEnv-v0", **kwargs)
        elif env_name == "CustomHalfCheetahBulletEnv":
            return gym.make("CustomHalfCheetahBulletEnv-v0", **kwargs)
        else:
            raise ValueError("Unknown environment name: ", env_name)

    @staticmethod
    def register(env_name, **kwargs):
        """Registers the specified environment, so that it can be instantiated
        by the RLLib algorithms by name.

        Args:
            env_name (str): name of the environment

        Returns:
            gym.env: the registered environment
        """
        env = EnvironmentFactory.create(env_name, **kwargs)
        register_env(env_name, lambda _: EnvironmentFactory.create(env_name, **kwargs))
        return env
