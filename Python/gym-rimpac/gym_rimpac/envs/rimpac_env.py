#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
from collections import namedtuple

import numpy as np
import gym
# from gym import spaces, error, utils
# from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
# from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

Observation = namedtuple('Observation',
                         ('decision_steps', 'terminal_steps'))


class RimpacEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        worker_id=0,
        base_port=None,
        seed=0,
        no_graphics=False
    ):
        file_name = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Build', 'Rimpac')
        self._env = UnityEnvironment(file_name, worker_id=worker_id, base_port=base_port, seed=seed, no_graphics=no_graphics)

        self._action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
        self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=(61,))

        self.steps = []
        self.observation = []

    def step(self, action):
        done, info = False, {}
        skip_frames = 4
        for _ in range(skip_frames):
            for team_id, (decision_steps, terminal_steps) in enumerate(self.steps):
                # if decision_steps.rewards.shape[0] == 0:
                #     action = np.zeros((0, 6))
                if terminal_steps.reward.shape[0] > 0:
                    done = True
                    info['win'] = int(terminal_steps.reward[0] == 1.0)

                for i, behavior_name in enumerate(self.behavior_names):
                    continuous_action = ActionTuple()
                    continuous_action.add_continuous(action[i][np.newaxis, :])
                    self._env.set_actions(behavior_name, continuous_action)
                # self._env.set_actions(behavior_name='Rimpac?team=1', action=action[0])
                # self._env.set_actions(behavior_name='Rimpac?team=2', action=action[1])

            if done:
                break

        if done:
            observation = np.array([obs.terminal_steps.obs for obs in self.observation])
            # if 0 in observation.shape:
            #     observation = self.observation_cache
            reward = np.array([obs.terminal_steps.reward for obs in self.observation])
            # if 0 in reward.shape:
            #     reward = np.zeros((1,))
        else:
            self._env.step()
            observation = self._update_environment_state()
            # if 0 in observation.shape:
            #     observation = self.observation_cache
            # self.observation_cache = observation
            reward = np.array([obs.decision_steps.reward for obs in self.observation])
            # if 0 in reward.shape:
            #     reward = np.zeros((1,))

            # reward -= 0.001

        return np.squeeze(observation, axis=1), np.squeeze(reward), done, info

    def reset(self):
        self._env.reset()
        self.behavior_names = [name for name in self._env.behavior_specs.keys()]
        print('RimpacEnv.behavior_names:', self.behavior_names)

        observation = self._update_environment_state()

        return np.squeeze(observation, axis=1)

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self._env.close()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _update_environment_state(self):
        self.steps = [self._env.get_steps(behavior_name=behavior) for behavior in self.behavior_names]
        self.observation = [Observation(*step) for step in self.steps]
        return np.array([obs.decision_steps.obs for obs in self.observation])
