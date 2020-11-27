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

    def __init__(self, n=9, worker_id=20800, name="BlackWhale"):
        self.env = UnityEnvironment(file_name=name, seed=1, worker_id=worker_id, side_channels=[EngineConfigurationChannel()])
        self.n = n

    def reset(self):
        self.env.reset()

        self.behavior_names = [name for name in self.env.behavior_specs.keys()]
        print('behavior_names:', self.behavior_names)

        self.__observe()

        observation = [obs.decision_steps.obs for obs in self.observation]
        # print('np.array(observation).shape:', np.array(observation).shape)

        return np.squeeze(np.array(observation), axis=1)

    def step(self, action):
        # https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md
        done = False
        # print('step.action:', action, len(action), len(action[0]))
        for team_id, (decision_steps, terminal_steps) in enumerate(self.steps):
            # print('team_id:', team_id, decision_steps.agent_id, terminal_steps.agent_id, type(terminal_steps.agent_id))
            # print('decision_teps:', decision_steps.agent_id)
            # print('terminal_steps:', terminal_steps.agent_id)
            if terminal_steps.agent_id.shape[0] > 0:
                print('terminal_steps.agent_id:', terminal_steps.agent_id)
                done = True
                action = np.zeros((0, 6))
            """
            for i, id_ in enumerate(decision_steps.agent_id):
                print('team_id: %d, enumerate(i:%d, id_: %d)' % (team_id, i, id_))
                # self.env.set_action_for_agent(behavior_name=BEHAVIOR_NAME.format(team=team_id), agent_id=id_, action=action[id_-1])
                self.env.set_action_for_agent(behavior_name=BEHAVIOR_NAME.format(team=team_id), agent_id=id_, action=action[id_, :])
            """
            self.env.set_actions(behavior_name=BEHAVIOR_NAME.format(team=team_id), action=action)
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
                reward = np.zeros((self.n, 1))
        else:
            observation = [obs.decision_steps.obs for obs in self.observation]
            observation = np.array(observation)
            if 0 in observation.shape:
                observation = self.observation_cache
            self.observation_cache = observation
            reward = [obs.decision_steps.reward for obs in self.observation]
            reward = np.array(reward)
            if 0 in reward.shape:
                reward = np.zeros((self.n, 1))

        return np.squeeze(observation, axis=1), np.squeeze(reward), done

    def close(self):
        self.env.close()

    def __observe(self):
        self.steps = [self.env.get_steps(behavior_name=name) for name in self.behavior_names]
        self.observation = [Observation(*step) for step in self.steps]
        return self.observation


if __name__ == "__main__":
    pass
