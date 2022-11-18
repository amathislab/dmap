import gym


gym.envs.registration.register(
    id="RandomPerturbationAntBulletEnv-v0",
    entry_point="pybullet_m.envs.ant:RandomPerturbationAntBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="RandomPerturbationWalker2DBulletEnv-v0",
    entry_point="pybullet_m.envs.walker:RandomPerturbationWalker2DBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="SymmetricRandomPerturbationWalker2DBulletEnv-v0",
    entry_point="pybullet_m.envs.walker:SymmetricRandomPerturbationWalker2DBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="RandomPerturbationHopperBulletEnv-v0",
    entry_point="pybullet_m.envs.hopper:RandomPerturbationHopperBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="RandomPerturbationHalfCheetahBulletEnv-v0",
    entry_point="pybullet_m.envs.half_cheetah:RandomPerturbationHalfCheetahBulletEnv",
    max_episode_steps=1000,
    reward_threshold=3000.0,
)

gym.envs.registration.register(
    id="RMAAntBulletEnv-v0",
    entry_point="pybullet_m.envs.ant:RMAAntBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="RMAWalker2DBulletEnv-v0",
    entry_point="pybullet_m.envs.walker:RMAWalker2DBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="RMASymmetricWalker2DBulletEnv-v0",
    entry_point="pybullet_m.envs.walker:RMASymmetricWalker2DBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="RMAHopperBulletEnv-v0",
    entry_point="pybullet_m.envs.hopper:RMAHopperBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="RMAHalfCheetahBulletEnv-v0",
    entry_point="pybullet_m.envs.half_cheetah:RMAHalfCheetahBulletEnv",
    max_episode_steps=1000,
    reward_threshold=3000.0,
)

gym.envs.registration.register(
    id="CustomAntBulletEnv-v0",
    entry_point="pybullet_m.envs.ant:CustomAntBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="CustomHopperBulletEnv-v0",
    entry_point="pybullet_m.envs.hopper:CustomHopperBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="CustomWalker2DBulletEnv-v0",
    entry_point="pybullet_m.envs.walker:CustomWalker2DBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.envs.registration.register(
    id="CustomHalfCheetahBulletEnv-v0",
    entry_point="pybullet_m.envs.half_cheetah:CustomHalfCheetahBulletEnv",
    max_episode_steps=1000,
    reward_threshold=3000.0,
)
