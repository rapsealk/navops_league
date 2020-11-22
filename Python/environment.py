#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
# from mlagents_envs.side_channel.

Observation = namedtuple('Observation',
                         ('decision_steps', 'terminal_steps'))

BEHAVIOR_NAME = "Warship?team={team}"


class UnityEnvironmentImpl:

    def __init__(self, name="BlackWhale"):
        self.env = UnityEnvironment(file_name=name, seed=1, side_channels=[EngineConfigurationChannel()])

    def reset(self):
        self.env.reset()

        self.behavior_names = [name for name in self.env.behavior_specs.keys()]

        self.__observe()

        observation = [obs.decision_steps.obs for obs in self.observation]
        # print('np.array(observation).shape:', np.array(observation).shape)

        return np.squeeze(np.array(observation), axis=1)

    def step(self, action):
        done = False
        for team_id, (decision_steps, terminal_steps) in enumerate(self.steps):
            if terminal_steps.agent_id:
                # print('terminal_steps.agent_id:', terminal_steps.agent_id)
                done = True
            for i, id_ in enumerate(decision_steps.agent_id):
                #if id_ in terminal_steps.agent_id:
                #    # dones[i // 9, i % 9] = 1.0
                #    dones[team_id, i] = 1.0
                #    continue    # TODO: train
                # dones[i // 9, i % 9] = 0.0
                #dones[team_id, i] = 0.0
                self.env.set_action_for_agent(behavior_name=BEHAVIOR_NAME.format(team=team_id), agent_id=id_, action=action[id_-1])
        self.env.step()
        self.__observe()

        if done:
            observation = [obs.terminal_steps.obs for obs in self.observation]
            observation = np.array(observation)
            if 0 in observation.shape:
                observation = self.observation_cache
            reward = [obs.terminal_steps.reward for obs in self.observation]
            reward = np.array(reward)
            if 0 in reward.shape:
                reward = np.zeros((2, 1))
        else:
            observation = [obs.decision_steps.obs for obs in self.observation]
            observation = np.array(observation)
            if 0 in observation.shape:
                observation = self.observation_cache
            self.observation_cache = observation
            reward = [obs.decision_steps.reward for obs in self.observation]
            reward = np.array(reward)
            if 0 in reward.shape:
                reward = np.zeros((2, 1))

        return np.squeeze(observation, axis=1), np.squeeze(reward), done

    def close(self):
        self.env.close()

    def __observe(self):
        self.steps = [self.env.get_steps(behavior_name=name) for name in self.behavior_names]
        self.observation = [Observation(*step) for step in self.steps]
        return self.observation


if __name__ == "__main__":
    pass
