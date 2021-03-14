#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import json
from datetime import datetime
from itertools import count
import threading
from threading import Thread, Lock
from multiprocessing import cpu_count

import gym
import gym_rimpac   # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import PPOAgent
from utils import SlackNotification, discount_rewards
from rating import EloRating

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='RimpacDiscrete-v0')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--worker-id', type=int, default=0)
parser.add_argument('--time-horizon', type=int, default=2048)
parser.add_argument('--seq_len', type=int, default=64)  # 0.1s per state-action
# parser.add_argument('--aggressive_factor', type=float, default=1.0)
# parser.add_argument('--defensive_factor', type=float, default=0.7)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--no-logging', action='store_true', default=False)
args = parser.parse_args()

# TODO: ML-Agents EventSideChannel(uuid.uuid4())

ENVIRONMENT = args.env
# Hyperparameters
TIME_HORIZON = args.time_horizon
SEQUENCE = args.seq_len
# AGGRESSIVE_FACTOR = args.aggressive_factor
# DEFENSIVE_FACTOR = args.defensive_factor
NO_LOGGING = args.no_logging

REWARD_FIELD = 2
HITPOINT_FIELD = -2
AMMO_FIELD = -14
FUEL_FIELD = -13
WORKERS = 6 # 4     # cpu_count()

if args.env == 'RimpacDiscreteSkipFrame-v0':
    AMMO_FIELD = -4
    FUEL_FIELD = -3


class Atomic:

    def __init__(self, dtype=int):
        self._value = dtype()
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            value = self._value
        return value


class Learner:

    def __init__(self):
        self._env = gym.make(ENVIRONMENT, no_graphics=args.no_graphics, worker_id=args.worker_id)
        self._target_agent = PPOAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            learning_rate=args.learning_rate
        )
        self._worker_agent = PPOAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            learning_rate=args.learning_rate
        )
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'pretrained', 'supervised.ckpt')
        if os.path.exists(checkpoint_path):
            try:
                self._target_agent.load(checkpoint_path)
            except RuntimeError:
                pass
        """
        self._opponent_agent = PPOAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n
        )
        """
        self._id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{ENVIRONMENT}'
        if not NO_LOGGING:
            self._writer = SummaryWriter(f'runs/{self._id}')

        self._training_episode = Atomic(int)
        self._lock = Lock()

    def run(self):
        threads = [
            Worker(self, self._training_episode, self._writer, args.worker_id+i+1)
            for i in range(WORKERS)
        ]
        for thread in threads:
            thread.start()

        observation_shape = self._env.observation_space.shape[0]
        results = []
        rate = 0.0
        ratings = (1200, 1200)
        for episode in count(1):
            rewards = []
            batch = []
            new_obs1, new_obs2 = self._env.reset()

            obs1 = np.concatenate([new_obs1] * SEQUENCE)
            obs2 = np.concatenate([new_obs2] * SEQUENCE)

            rnn_output_size = self._target_agent.rnn_output_size
            h_out = [(torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float)),
                     (torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float))]

            done = False

            self._worker_agent.set_state_dict(
                self._target_agent.state_dict()
            )

            while not done:
                # for t in range(TIME_HORIZON):
                h_in = h_out.copy()
                action1, prob1, h_out[0] = self._worker_agent.get_action(obs1, h_in[0])
                # action2, prob2, h_out[1] = self._opponent_agent.get_action(obs2, h_in[1])
                action2 = np.random.randint(self._env.action_space.n)
                action = np.array([action1, action2], dtype=np.uint8)

                next_obs, reward, done, info = self._env.step(action)

                rewards.append(reward[0])

                next_obs1 = np.concatenate((obs1[observation_shape:], next_obs[0]))
                next_obs2 = np.concatenate((obs2[observation_shape:], next_obs[1]))

                # self._target_agent.append((obs1, action[0], reward1, next_obs1, prob1, h_in[0], h_out[0], not done))
                batch.append((obs1, action[0], reward[0], next_obs1, prob1, h_in[0], h_out[0], not done))
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
                    if not NO_LOGGING:
                        self._writer.add_scalar('r/rating', ratings[0], episode)
                        self._writer.add_scalar('r/rewards', np.sum(rewards), episode)
                    rewards = discount_rewards(rewards)

                    for traj, r in zip(batch, rewards):
                        self._worker_agent.append(traj[:2] + (r,) + traj[3:])

                    # Logging
                    hitpoint = obs1[HITPOINT_FIELD]
                    damage = 1 - obs2[HITPOINT_FIELD]
                    ammo = 1 - obs1[AMMO_FIELD]
                    fuel = 1 - obs1[FUEL_FIELD]
                    # TODO: DPS
                    if not NO_LOGGING:
                        self._writer.add_scalar('r/hitpoint', hitpoint, episode)
                        self._writer.add_scalar('r/hitpoint_gap', obs1[HITPOINT_FIELD] - obs2[HITPOINT_FIELD], episode)
                        self._writer.add_scalar('r/damage', damage, episode)
                        self._writer.add_scalar('r/resource/ammo_usage', ammo, episode)
                        self._writer.add_scalar('r/resource/fuel_usage', fuel, episode)
                    break

                obs1, obs2 = next_obs1, next_obs2

            time_horizons = self._training_episode.increment()
            with self._lock:
                # loss, pi_loss, value_loss = self._target_agent.train(TIME_HORIZON)
                loss, pi_loss, value_loss = self._target_agent.apply_gradient(self._worker_agent, time_horizon=TIME_HORIZON)
            if not NO_LOGGING:
                self._writer.add_scalar('loss/total', loss, time_horizons)
                self._writer.add_scalar('loss/policy', pi_loss, time_horizons)
                self._writer.add_scalar('loss/value', value_loss, time_horizons)

            if episode % 100 == 0:
                new_rate = np.mean(results)
                if not NO_LOGGING:
                    self._writer.add_scalar('r/result', new_rate, episode // 100)
                results.clear()
                if rate < new_rate:
                    with self._lock:
                        self._target_agent.save(os.path.join(os.path.dirname(__file__), 'checkpoints', f'{self._id}-r{new_rate}.ckpt'))
                    rate = new_rate

                if episode % 1000 == 0:
                    with self._lock:
                        self._target_agent.save(os.path.join(os.path.dirname(__file__), 'checkpoints', f'{self._id}-e{episode}-r{new_rate}.ckpt'))

        for thread in threads:
            thread.join()


class Worker(Thread):

    def __init__(self, learner, training_episode, writer, worker_id=1):
        Thread.__init__(self, daemon=True)
        print(f'[{datetime.now().isoformat()}] Thread({threading.get_ident()})')
        self._env = gym.make(ENVIRONMENT, no_graphics=True, worker_id=worker_id)
        self._worker_agent = PPOAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            learning_rate=args.learning_rate
        )
        self._learner = learner
        self._training_episode = training_episode
        self._writer = writer
        """
        self._opponent_agent = PPOAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n
        )
        """

    def run(self):
        while True:
            self.load_learner_parameters()
            observation_shape = self._env.observation_space.shape[0]
            while True:
                rewards = []
                batch = []
                new_obs1, new_obs2 = self._env.reset()

                obs1 = np.concatenate([new_obs1] * SEQUENCE)
                obs2 = np.concatenate([new_obs2] * SEQUENCE)

                rnn_output_size = self._worker_agent.rnn_output_size
                h_out = [(torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                          torch.zeros([1, 1, rnn_output_size], dtype=torch.float)),
                         (torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                          torch.zeros([1, 1, rnn_output_size], dtype=torch.float))]

                done = False

                while not done:
                    # for t in range(TIME_HORIZON):
                    h_in = h_out.copy()
                    action1, prob1, h_out[0] = self._worker_agent.get_action(obs1, h_in[0])
                    # action2, prob2, h_out[1] = self._opponent_agent.get_action(obs2, h_in[1])
                    action2 = np.random.randint(self._env.action_space.n)
                    action = np.array([action1, action2], dtype=np.uint8)

                    next_obs, reward, done, info = self._env.step(action)

                    rewards.append(reward[0])

                    next_obs1 = np.concatenate((obs1[observation_shape:], next_obs[0]))
                    next_obs2 = np.concatenate((obs2[observation_shape:], next_obs[1]))

                    # self._target_agent.append((obs1, action[0], reward1, next_obs1, prob1, h_in[0], h_out[0], not done))
                    batch.append((obs1, action[0], reward[0], next_obs1, prob1, h_in[0], h_out[0], not done))
                    # self._opponent_agent.append((obs2, action[1], reward2, next_obs2, prob2, h_in[1], h_out[1], not done))

                    if done:
                        rewards = discount_rewards(rewards)

                        for traj, r in zip(batch, rewards):
                            self._worker_agent.append(traj[:2] + (r,) + traj[3:])
                        break

                    obs1, obs2 = next_obs1, next_obs2

                time_horizons = self._training_episode.increment()
                with self._learner._lock:
                    loss, pi_loss, value_loss = self._learner._target_agent.apply_gradient(self._worker_agent, time_horizon=TIME_HORIZON)
                if not NO_LOGGING:
                    self._writer.add_scalar('loss/total', loss, time_horizons)
                    self._writer.add_scalar('loss/policy', pi_loss, time_horizons)
                    self._writer.add_scalar('loss/value', value_loss, time_horizons)
                # print(f'[{datetime.now().isoformat()}] Episode #{time_horizons}: Loss({loss}, {pi_loss}, {value_loss})')
                # _ = self._opponent_agent.train()

    def load_learner_parameters(self):
        self._worker_agent.set_state_dict(
            self._learner._target_agent.state_dict()
        )


# @SlackNotification(SLACK_API_TOKEN)
def main():
    Learner().run()


if __name__ == "__main__":
    main()
