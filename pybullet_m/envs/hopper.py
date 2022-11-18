import os
import numpy as np
import time
import random
import gym
from pybullet_envs.robot_locomotors import Hopper, WalkerBase
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from definitions import ROOT_DIR
from pybullet_m.envs.mixins import HistoryMixin
from pybullet_m.envs.walker import CustomWalker2DBulletEnv
from pybullet_m.helpers.xml_generator import perturb_hopper_xml


class HopperXml(Hopper):
    """Replica of the Hopper robot of PyBullet, but enabling the uses to select the xml
    containing the specifications of the robot
    """

    def __init__(self, xml_path, action_dim=3, obs_dim=15):
        """Init function

        Args:
            xml_path (str): path to the Hopper xml file
            action_dim (int, optional): Action size. Defaults to 3.
            obs_dim (int, optional): Observation size. Defaults to 15.
        """
        xml_path = os.path.join(ROOT_DIR, xml_path)
        WalkerBase.__init__(
            self, xml_path, "torso", action_dim=action_dim, obs_dim=obs_dim, power=0.75
        )
        self.np_random, _ = gym.utils.seeding.np_random()

    def alive_bonus(self, z, pitch):
        # 0.64 because in the unperturbed env the initial z is 1.25 and the threshold is 0.8
        return +1 if z > 0.64 * self.initial_z and abs(pitch) < 1.0 else -1


class CustomHopperBulletEnv(WalkerBaseBulletEnv):
    """Replica of the Hopper environment of PyBullet, but enabling the uses
    to select the xml containing the specifications of the robot
    """

    def __init__(self, xml_path, render=False):
        """Init function

        Args:
            xml_path (str): path to the Hopper xml file
            render (bool, optional): whether to output the state to a video. Defaults to False.
        """
        self.robot = HopperXml(xml_path)
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


class RandomPerturbationHopperBulletEnv(CustomHopperBulletEnv):
    """Hopper environment selecting a random perturbation at the beginning of each episode.
    It does not include the raw perturbation in the state, as it is the environment used
    by Simple
    """

    def __init__(
        self, sigma, render=False, action_dim=3, obs_dim=15, perturbation_vals=None
    ):
        """Init function

        Args:
            sigma (float): intensity of the perturbation, in range (0, 1)
            render (bool, optional): whether to output the state to a video. Defaults to False.
            action_dim (int, optional): Action size. Defaults to 3.
            obs_dim (int, optional): Observation size. Defaults to 15.
            perturbation_vals (list, optional): If provided, fixes the perturbation to those
            values. Must be an iterable of size 5. Defaults to None.
        """
        self.name = "RandomPerturbationHopperBulletEnv"
        self.perturbation_list = [
            "torso_size_perturb",
            "leg_size_perturb",
            "leg_length_perturb",
            "foot_size_perturb",
            "foot_length_perturb",
        ]
        self.sigma = sigma
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.temp_dir = os.path.join(ROOT_DIR, "pybullet_m", "xmls", "hopper", "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.base_xml_path = os.path.join(
            ROOT_DIR, "pybullet_m", "xmls", "hopper", "hopper.xml"
        )
        CustomHopperBulletEnv.__init__(self, self.base_xml_path, render)
        self.already_reset = False
        if perturbation_vals is None:
            self.random_perturb = True
            self.current_perturb = self.make_perturb_from_vals(
                np.zeros(len(self.perturbation_list))
            )
        else:
            assert len(self.perturbation_list) == len(perturbation_vals)
            self.random_perturb = False
            self.current_perturb = self.make_perturb_from_vals(perturbation_vals)

    def reset(self):
        """Reset the environment state at the beginning of the episode

        Returns:
            np.array: initial state
        """
        if self.already_reset:
            self._p.removeBody(1)
        else:
            self.already_reset = True
        xml_file_name = f"{os.getpid()}_{time.time()}.xml"
        if self.random_perturb:
            self.current_perturb = self.make_random_perturb()
        xml_file_path = os.path.join(self.temp_dir, xml_file_name)
        perturb_hopper_xml(self.base_xml_path, xml_file_path, **self.current_perturb)

        self.robot = HopperXml(xml_file_path, self.action_dim, self.obs_dim)
        self.stateId = -1
        state = CustomHopperBulletEnv.reset(self)
        os.remove(xml_file_path)
        return state

    def step(self, action):
        """Advances the simulation by one time step.

        Args:
            action (np.array): Torques to apply to the 3 joints of the Hopper

        Returns:
            tuple(np.array, float, bool, dict): elements of the transition
        """
        state, reward, done, info = CustomWalker2DBulletEnv.step(self, action)
        info.update(self.current_perturb)
        return state, reward, done, info

    def make_random_perturb(self):
        return {
            p: random.uniform(-self.sigma, self.sigma) for p in self.perturbation_list
        }

    def make_perturb_from_vals(self, perturbation_vals):
        return {
            p: self.sigma * val
            for p, val in zip(self.perturbation_list, perturbation_vals)
        }


class RMAHopperBulletEnv(RandomPerturbationHopperBulletEnv, HistoryMixin):
    """Hopper environment selecting a random perturbation at the beginning of each episode.
    It includes the raw perturbation in the state, so that the Oracle agent can use it. It also
    optionally includes a history of transitions, to be used by RMA, TCN and DMAP.

    N.B.: RMA, TCN and DMAP ignore the raw environment perturbation, as they are trained to
    adapt based on the transition history.
    """

    def __init__(
        self,
        sigma,
        render=False,
        include_adapt_state=False,
        num_memory_steps=30,  # Number of previous transitions to include in the state - only works if include_adapt_state=True
        perturbation_vals=None,
    ):
        """Init function

        Args:
            sigma (float): intensity of the perturbation, in range (0, 1)
            render (bool, optional): whether to output the state to a video. Defaults to False.
            include_adapt_state (bool, optional): whether to return a sequence of past states
            and actions together with the state. Defaults to False.
            num_memory_steps (int, optional): if include_adapt_state is True, specifies how many
            past states and actions to include. Defaults to 30.
        """
        super().__init__(sigma, render, perturbation_vals=perturbation_vals)
        self._init_addon(include_adapt_state, num_memory_steps)

    def reset(self):
        """Reset the environment state at the beginning of the episode

        Returns:
            np.array: initial state
        """
        state = super().reset()
        return self.create_rma_reset_state(state)

    def step(self, action):
        """Advances the simulation by one time step.

        Args:
            action (np.array): Torques to apply to the 3 joints of the Hopper

        Returns:
            tuple(np.array, float, bool, dict): elements of the transition
        """
        state, reward, done, info = super().step(action)
        return_dict = self.create_rma_step_state(state, action)
        return return_dict, reward, done, info
