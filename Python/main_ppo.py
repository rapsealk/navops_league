#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import json
from datetime import datetime
from itertools import count

import gym
import gym_rimpac   # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import PPOAgent
from utils import SlackNotification
from rating import EloRating

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='RimpacDiscrete-v0')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--time-horizon', type=int, default=128)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--aggressive_factor', type=float, default=1.0)
parser.add_argument('--defensive_factor', type=float, default=0.7)
args = parser.parse_args()

# TODO: ML-Agents EventSideChannel(uuid.uuid4())


ENVIRONMENT = args.env
# Hyperparameters
TIME_HORIZON = args.time_horizon
SEQUENCE = args.seq_len
AGGRESSIVE_FACTOR = args.aggressive_factor
DEFENSIVE_FACTOR = args.defensive_factor

REWARD_FIELD = 2
HITPOINT_FIELD = -2
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


def shape_rewards(observation, next_observation, winner):
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
    if winner is not None:
        rewards[0] += 10 * (winner == 0) - 10 * (winner != 0)
        rewards[1] += 10 * (winner == 1) - 10 * (winner != 1)
    # reward[0] = hp_change[0] - hp_change[1] + 10 * (info.get('win', -1) == 0)
    # reward[1] = hp_change[1] - hp_change[0] + 10 * (info.get('win', -1) == 1)
    return rewards


class Learner:

    def __init__(self):
        self._env = gym.make(ENVIRONMENT, no_graphics=args.no_graphics)
        self._target_agent = PPOAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n
        )
        """
        self._opponent_agent = PPOAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n
        )
        """
        self._writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{ENVIRONMENT}')

    def run(self):
        observation_shape = self._env.observation_space.shape[0]
        results = []
        ratings = (1200, 1200)
        time_horizons = 0
        # best_rating = 1200
        for episode in count(1):
            rewards = []
            new_obs1, new_obs2 = self._env.reset()

            # obs1 = np.zeros((observation_shape * SEQUENCE,))
            # obs1 = np.concatenate((obs1[observation_shape:], new_obs1))
            obs1 = np.concatenate([new_obs1] * SEQUENCE)
            # obs2 = np.zeros((observation_shape * SEQUENCE,))
            # obs2 = np.concatenate((obs1[observation_shape:], new_obs2))
            obs2 = np.concatenate([new_obs2] * SEQUENCE)

            rnn_output_size = self._target_agent.rnn_output_size
            """
            h_out = [(np.zeros((1, 1, rnn_output_size)),
                      np.zeros((1, 1, rnn_output_size))),
                     (np.zeros((1, 1, rnn_output_size)),
                      np.zeros((1, 1, rnn_output_size)))]
            """
            h_out = [(torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float)),
                     (torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float))]

            done = False

            while not done:
                for t in range(TIME_HORIZON):
                    # print(f'obs1: {obs1[-observation_shape:]}')
                    h_in = h_out.copy()
                    action1, prob1, h_out[0] = self._target_agent.get_action(obs1, h_in[0])
                    # action2, prob2, h_out[1] = self._opponent_agent.get_action(obs2, h_in[1])
                    action2 = np.random.randint(self._env.action_space.n)
                    action = np.array([action1, action2], dtype=np.uint8)

                    next_obs, reward, done, info = self._env.step(action)

                    reward1, reward2 = shape_rewards((obs1, obs2), next_obs, info.get('win', None))
                    rewards.append(reward1)

                    next_obs1 = np.concatenate((obs1[observation_shape:], next_obs[0]))
                    next_obs2 = np.concatenate((obs2[observation_shape:], next_obs[1]))

                    self._target_agent.append((obs1, action[0], reward1, next_obs1, prob1, h_in[0], h_out[0], not done))
                    # self._opponent_agent.append((obs2, action[1], reward2, next_obs2, prob2, h_in[1], h_out[1], not done))

                    if done:
                        print(f'[{datetime.now().isoformat()}] Done! ({obs1[HITPOINT_FIELD]}, {obs2[HITPOINT_FIELD]}) -> {info.get("win", None)}')
                        winner = info.get('win', -1)
                        if winner == -1:    # Draw
                            results.append(False)
                        else:
                            a_win = bool(1 - winner)
                            results.append(a_win)
                            ratings = EloRating.calc(ratings[0], ratings[1], a_win)
                        self._writer.add_scalar('r/rating', ratings[0], episode)
                        rewards = discount_rewards(rewards)
                        self._writer.add_scalar('r/rewards', np.mean(rewards), episode)

                        # Logging
                        hitpoint = obs1[HITPOINT_FIELD]
                        damage = 1 - obs2[HITPOINT_FIELD]
                        ammo = 1 - obs1[AMMO_FIELD]
                        fuel = 1 - obs1[FUEL_FIELD]
                        # TODO: DPS
                        self._writer.add_scalar('r/hitpoint', hitpoint, episode)
                        self._writer.add_scalar('r/hitpoint_gap', obs1[HITPOINT_FIELD] - obs2[HITPOINT_FIELD], episode)
                        self._writer.add_scalar('r/damage', damage, episode)
                        self._writer.add_scalar('r/resource/ammo_usage', ammo, episode)
                        self._writer.add_scalar('r/resource/fuel_usage', fuel, episode)
                        break

                    obs1, obs2 = next_obs1, next_obs2

                loss = self._target_agent.train()
                time_horizons += 1
                self._writer.add_scalar('loss', loss, time_horizons)
                print(f'[{datetime.now().isoformat()}] Episode #{time_horizons}: Loss({loss})')
                # _ = self._opponent_agent.train()

            if episode % 100 == 0:
                self._writer.add_scalar('r/result', np.mean(results), episode // 100)
                results.clear()


# @SlackNotification(SLACK_API_TOKEN)
def main():
    Learner().run()


if __name__ == "__main__":
    main()
