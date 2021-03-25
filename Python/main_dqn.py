#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import json
from datetime import datetime
from itertools import count
from threading import Thread
from multiprocessing import cpu_count

import gym
import gym_navops   # noqa: F401
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import DQNAgent, DDQNAgent
from memory import ReplayBuffer
from utils import SlackNotification
from rating import EloRating

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsDiscrete-v0')
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--aggressive_factor', type=float, default=1.0)
parser.add_argument('--defensive_factor', type=float, default=0.7)
args = parser.parse_args()


ENVIRONMENT = args.env
# Hyperparameters
BATCH_SIZE = args.batch_size
SEQUENCE = args.seq_len
AGGRESSIVE_FACTOR = args.aggressive_factor
DEFENSIVE_FACTOR = args.defensive_factor

EPSILON_INITIAL_VALUE = 1.0
EPSILON_DISCOUNT = 0.99
EPSILON_MINIMUM = 0.01

REWARD_FIELD = 2
HITPOINT_FIELD = -1
AMMO_FIELD = -4
FUEL_FIELD = -3


def discount_rewards(rewards: np.ndarray, gamma=0.998):  # 347 -> 0.5
    discounted = np.zeros_like(rewards)
    discounted[-1] = rewards[-1]
    n = len(rewards) - 1
    for i in range(n):
        discounted[n-i-1] = discounted[n-i] * gamma + rewards[n-i-1]
    return discounted


def get_distance(a: np.ndarray, b: np.ndarray):
    return np.sqrt(np.sum(np.power(a - b, 2)))


def get_distance_reward(a: np.ndarray, b: np.ndarray):
    return 1 / (np.power(1 + get_distance(a, b), 2) * 100)


def shape_rewards(observation, next_observation):
    # Reward shaping
    # - Blade & Soul (https://arxiv.org/abs/1904.03821)
    hp_change = [
        next_observation[0][HITPOINT_FIELD] - observation[0][HITPOINT_FIELD],
        next_observation[1][HITPOINT_FIELD] - observation[1][HITPOINT_FIELD]
    ]
    rewards = np.array([
        (DEFENSIVE_FACTOR * hp_change[0] - AGGRESSIVE_FACTOR * hp_change[1]),
        (DEFENSIVE_FACTOR * hp_change[1] - AGGRESSIVE_FACTOR * hp_change[0])
    ]) + get_distance_reward(next_observation[0][:2], next_observation[1][:2])
    # reward[0] = hp_change[0] - hp_change[1] + 10 * (info.get('win', -1) == 0)
    # reward[1] = hp_change[1] - hp_change[0] + 10 * (info.get('win', -1) == 1)
    return rewards


class Learner:

    def __init__(self):
        self._env = gym.make(ENVIRONMENT)
        self._target_agent = DDQNAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            BATCH_SIZE
        )
        self._opponent_agent = DDQNAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            BATCH_SIZE
        )
        self._buffer = ReplayBuffer()
        self._writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{ENVIRONMENT}')

        self._num_workers = cpu_count() - 2

    def run(self):
        workers = [
            Worker(self._target_agent, self._buffer, i+1)
            for i in range(self._num_workers)
        ]
        for worker in workers:
            worker.start()

        self.validate()

        for worker in workers:
            worker.join()

    def validate(self):
        observation_shape = self._env.observation_space.shape[0]
        results = []
        ratings = (1200, 1200)
        best_rating = 1200
        for episode in count(1):
            rewards = []
            new_obs1, new_obs2 = self._env.reset()

            obs1 = np.zeros((observation_shape * SEQUENCE,))
            obs1 = np.concatenate((obs1[observation_shape:], new_obs1))
            obs2 = np.zeros((observation_shape * SEQUENCE,))
            obs2 = np.concatenate((obs1[observation_shape:], new_obs2))

            for timestep in count(1):
                action1 = self._opponent_agent.get_action(obs1)
                action2 = self._target_agent.get_action(obs2)
                action = np.array([action1, action2], dtype=np.uint8)

                next_obs, reward, done, info = self._env.step(action)

                _, reward = shape_rewards((obs1, obs2), next_obs)
                rewards.append(reward)

                next_obs1 = np.concatenate((obs1[observation_shape:], next_obs[0]))
                next_obs2 = np.concatenate((obs2[observation_shape:], next_obs[1]))

                obs1, obs2 = next_obs1, next_obs2

                if done:
                    winner = info.get('win', 0)
                    if winner == -1:    # Draw
                        results.append(False)
                    else:
                        a_win = bool(1 - winner)
                        results.append(not a_win)
                        ratings = EloRating.calc(ratings[0], ratings[1], a_win)
                    self._writer.add_scalar('r/rating', ratings[1], episode)

                    rewards = discount_rewards(rewards)
                    self._writer.add_scalar('r/rewards', np.mean(rewards), episode)

                    if ratings[1] > best_rating:
                        self._target_agent.save(
                            os.path.join(
                                os.path.dirname(__file__),
                                'checkpoints',
                                f'navops-discrete-v0-dqn-{episode:05}.ckpt'
                            )
                        )
                        best_rating = ratings[1]

                    if len(self._buffer) >= BATCH_SIZE:
                        loss = self._target_agent.train(self._buffer)
                        self._writer.add_scalar('loss/Q', loss, episode)

                    if episode % 100 == 0:
                        self._writer.add_scalar('r/result', np.mean(results), episode // 100)
                        results.clear()

                    if episode % 200 == 0:
                        self._opponent_agent.set_state_dict(self._target_agent.state_dict())
                        ratings = (ratings[1], ratings[1])

                    # Logging
                    hitpoint = obs2[HITPOINT_FIELD]
                    damage = 1 - obs1[HITPOINT_FIELD]
                    ammo = 1 - obs2[AMMO_FIELD]
                    fuel = 1 - obs2[FUEL_FIELD]
                    self._writer.add_scalar('r/timestep', timestep, episode)
                    # TODO: DPS
                    self._writer.add_scalar('r/hitpoint', hitpoint, episode)
                    self._writer.add_scalar('r/hitpoint_gap', obs2[HITPOINT_FIELD] - obs1[HITPOINT_FIELD], episode)
                    self._writer.add_scalar('r/damage', damage, episode)
                    self._writer.add_scalar('r/ammo', ammo, episode)
                    self._writer.add_scalar('r/fuel', fuel, episode)

                    break


class Worker(Thread):

    def __init__(
        self,
        target_agent,
        buffer,
        worker_id=0
    ):
        Thread.__init__(self, daemon=True)
        self._worker_id = worker_id
        self._target_agent = target_agent
        self._buffer = buffer

        self._env = gym.make(
            ENVIRONMENT,
            worker_id=worker_id,
            no_graphics=True
        )
        self._agent1 = DDQNAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            BATCH_SIZE,
            force_cpu=True
        )
        self._agent2 = DDQNAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            BATCH_SIZE,
            force_cpu=True
        )

    def run(self):
        observation_shape = self._env.observation_space.shape[0]
        epsilon = EPSILON_INITIAL_VALUE
        for episode in count(1):
            new_obs1, new_obs2 = self._env.reset()

            obs1 = np.zeros((observation_shape * SEQUENCE,))
            obs1 = np.concatenate((obs1[observation_shape:], new_obs1))
            obs2 = np.zeros((observation_shape * SEQUENCE,))
            obs2 = np.concatenate((obs1[observation_shape:], new_obs2))

            state_dict = self._target_agent.state_dict()
            self._agent1.set_state_dict(state_dict)
            self._agent2.set_state_dict(state_dict)

            buffer1, buffer2 = [], []

            while True:
                if epsilon > np.random.random():
                    action1 = np.random.randint(self._env.action_space.n)
                    action2 = np.random.randint(self._env.action_space.n)
                else:
                    action1 = self._agent1.get_action(obs1)
                    action2 = self._agent2.get_action(obs2)
                action = np.array([action1, action2], dtype=np.uint8)

                next_obs, reward, done, info = self._env.step(action)

                reward = shape_rewards((obs1, obs2), next_obs)

                next_obs1 = np.concatenate((obs1[observation_shape:], next_obs[0]))
                next_obs2 = np.concatenate((obs2[observation_shape:], next_obs[1]))

                buffer1.append((obs1, action1, reward[0], next_obs1, done))
                buffer2.append((obs2, action2, reward[1], next_obs2, done))

                obs1, obs2 = next_obs1, next_obs2

                if done:
                    trajectory1 = np.stack(buffer1)
                    trajectory2 = np.stack(buffer2)

                    trajectory1[:, REWARD_FIELD] = discount_rewards(trajectory1[:, REWARD_FIELD])
                    trajectory2[:, REWARD_FIELD] = discount_rewards(trajectory2[:, REWARD_FIELD])
                    for s, a, r, s_, d in np.concatenate([trajectory1, trajectory2]):
                        self._buffer.push(s, a, r, s_, d)

                    epsilon = max(epsilon * EPSILON_DISCOUNT, EPSILON_MINIMUM)

                    break


@SlackNotification(SLACK_API_TOKEN)
def main():
    Learner().run()


if __name__ == "__main__":
    main()
