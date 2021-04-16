#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import random
from collections import deque
from itertools import count

import numpy as np

import gym
import gym_navops   # noqa: F401
from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v2')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--n', type=int, default=3)
# parser.add_argument('--worker-id', type=int, default=0)
# parser.add_argument('--time-horizon', type=int, default=2048)
# parser.add_argument('--seq_len', type=int, default=64)  # 0.1s per state-action
# parser.add_argument('--learning-rate', type=float, default=1e-3)
# parser.add_argument('--no-logging', action='store_true', default=False)
args = parser.parse_args()


class ReplayBuffer:

    def __init__(self, capacity=1_000_000, seed=0):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, *args):
        self._buffer.append(args)

    def sample(self, batch_size, on_policy=False):
        batch_size = min(batch_size, len(self._buffer))
        if on_policy:
            return [self._buffer[-1]]
        return random.sample(self._buffer, batch_size)

    def __len__(self):
        return len(self._buffer)


def main():
    build_path = os.path.join('/', 'Users', 'rapsealk', 'Desktop', 'NavOps-v2')
    # build_path = Path('/') / 'Users' / 'rapsealk' / 'Desktop' / 'NavOps-v2'
    env = gym.make(args.env, no_graphics=args.no_graphics, worker_id=0, override_path=build_path)

    observations = env.reset()

    agents = [Agent(env.observation_space.shape[0], np.sum(env.action_space.nvec), agent_id=i, n=args.n)
              for i in range(args.n)]
    buffer = ReplayBuffer()

    for episode in count(1):
        for step in count(1):
            actions = []
            for agent, observation in zip(agents, observations):
                actions.append(action := agent.select_action(observation))
                # print(f'[MAIN] STEP={step} AGENT={agent.agent_id} ACTION={action}')
            actions = np.asarray(actions)   # , dtype=np.uint8)
            actions_mh = np.asarray([actions, actions]).T

            next_observations, rewards, done, info = env.step(actions_mh)
            # print(f'[MAIN] episode({episode}) step({step})')
            # print(f'[MAIN] - next_observations: {next_observations.shape} (hp: {",".join(map(str, next_observations[:, 0]))})')
            # print(f'[MAIN] - rewards: {rewards.shape}')
            # print(f'[MAIN] - done: {done}')
            # print(f'[MAIN] - info: {info}')

            buffer.push(observations, actions, next_observations, rewards, done)

            observations = next_observations

            if len(buffer) > 5 and len(buffer) % 5 == 0:
                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    batch = buffer.sample(args.batch_size)
                    agent.learn(batch, other_agents)
                # break

        # if episode == 2:
        break

    env.close()


if __name__ == "__main__":
    main()
