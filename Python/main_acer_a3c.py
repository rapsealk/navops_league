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

from models.pytorch_impl import AcerAgent, LstmActorCriticModel
from memory import ReplayBuffer
from utils import SlackNotification, discount_rewards, Atomic
from rating import EloRating

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='RimpacDiscrete-v0')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--worker-id', type=int, default=0)
parser.add_argument('--buffer-size', type=int, default=1000000)
parser.add_argument('--time-horizon', type=int, default=32)   # 2048
parser.add_argument('--seq-len', type=int, default=64)  # 0.1s per state-action
# parser.add_argument('--aggressive_factor', type=float, default=1.0)
# parser.add_argument('--defensive_factor', type=float, default=0.7)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--no-logging', action='store_true', default=False)
args = parser.parse_args()

# TODO: ML-Agents EventSideChannel(uuid.uuid4())

environment = args.env
# Hyperparameters
rollout = args.time_horizon
batch_size = 4
sequence_length = args.seq_len
# AGGRESSIVE_FACTOR = args.aggressive_factor
# DEFENSIVE_FACTOR = args.defensive_factor
learning_rate = args.learning_rate
no_logging = args.no_logging

field_hitpoint = -2
field_ammo = -14
field_fuel = -13
workers = cpu_count()

if args.env == 'RimpacDiscreteSkipFrame-v0':
    field_ammo = -4
    field_fuel = -3


class Learner:

    def __init__(self):
        self._env = gym.make(environment, no_graphics=args.no_graphics, worker_id=args.worker_id)
        self._buffer = ReplayBuffer(args.buffer_size)
        self._target_model = LstmActorCriticModel(
            self._env.observation_space.shape[0] * sequence_length,
            self._env.action_space.n,
            hidden_size=256
        )
        self._target_agent = AcerAgent(
            model=self._target_model,
            buffer=self._buffer,
            learning_rate=learning_rate,
            cuda=True
        )
        self._worker_agent = AcerAgent(
            model=self._target_model,
            buffer=self._buffer,
            learning_rate=learning_rate,
            cuda=True
        )
        """Checkpoints
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'pretrained', 'supervised.ckpt')
        if os.path.exists(checkpoint_path):
            try:
                self._target_agent.load(checkpoint_path)
            except RuntimeError:
                pass
        """
        """
        self._opponent_agent = PPOAgent(
            self._env.observation_space.shape[0] * sequence_length,
            self._env.action_space.n
        )
        """
        self._id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{environment}'
        if not no_logging:
            self._writer = SummaryWriter(f'runs/{self._id}')

        self._training_episode = Atomic(int)
        self._lock = Lock()

    def run(self):
        threads = [
            Worker(self, self._buffer, self._training_episode, self._writer, args.worker_id+i+1)
            for i in range(workers)
        ]
        for thread in threads:
            thread.start()

        observation_shape = self._env.observation_space.shape[0]
        result_wins = []
        result_draws = []
        result_loses = []
        ratings = (1200, 1200)
        training_step = 0
        for episode in count(1):
            rewards = []
            new_obs1, new_obs2 = self._env.reset()

            obs1 = np.concatenate([new_obs1] * sequence_length)
            obs2 = np.concatenate([new_obs2] * sequence_length)

            rnn_output_size = self._target_agent.rnn_output_size
            h_out = [(torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float)),
                     (torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float))]

            done = False

            """
            self._worker_agent.set_state_dict(
                self._target_agent.state_dict()
            )
            """

            while not done:
                batch = []
                for t in range(rollout):
                    h_in = h_out.copy()
                    action1, prob1, h_out[0] = self._target_agent.get_action(obs1, h_in[0])
                    # action2, prob2, h_out[1] = self._opponent_agent.get_action(obs2, h_in[1])
                    action2 = np.random.randint(self._env.action_space.n)
                    action = np.array([action1, action2], dtype=np.uint8)

                    next_obs, reward, done, info = self._env.step(action)

                    rewards.append(reward[0])

                    next_obs1 = np.concatenate((obs1[observation_shape:], next_obs[0]))
                    next_obs2 = np.concatenate((obs2[observation_shape:], next_obs[1]))

                    batch.append((obs1, action[0], reward[0], next_obs1, prob1, h_in[0], h_out[0], not done))

                    if done:
                        print(f'[{datetime.now().isoformat()}] Done! ({obs1[field_hitpoint]}, {obs2[field_hitpoint]}) -> {info.get("win", None)}')

                        result_wins.append(info.get('win', -1) == 0)
                        result_loses.append(info.get('win', -1) == 1)
                        result_draws.append(info.get('win', -1) == -1)

                        ratings = EloRating.calc(ratings[0], ratings[1], info.get('win', -1) == 0)
                        if not no_logging:
                            self._writer.add_scalar('r/rewards', np.sum(rewards), episode)
                            self._writer.add_scalar('r/rating', ratings[0], episode)
                            self._writer.add_scalar('logging/hitpoint', obs1[field_hitpoint], episode)
                            self._writer.add_scalar('logging/hitpoint_gap', obs1[field_hitpoint] - obs2[field_hitpoint], episode)
                            self._writer.add_scalar('logging/damage', 1 - obs2[field_hitpoint], episode)
                            self._writer.add_scalar('logging/ammo_usage', 1 - obs1[field_ammo], episode)
                            self._writer.add_scalar('logging/fuel_usage', 1 - obs1[field_fuel], episode)
                            if episode % 100 == 0:
                                # TODO: matplotlib stacked percentage bar
                                self._writer.add_scalar('r/wins', np.mean(result_wins), episode)
                                self._writer.add_scalar('r/loses', np.mean(result_loses), episode)
                                self._writer.add_scalar('r/draws', np.mean(result_draws), episode)
                        break

                    obs1, obs2 = next_obs1, next_obs2

                self._buffer.push(batch)
                if len(self._buffer) > 500:
                    training_step += 1
                    loss = self._target_agent.train(batch_size, on_policy=True)
                    loss += self._target_agent.train(batch_size)
                    print(f'[{datetime.now().isoformat()}] Loss: {loss}')
                    if not no_logging:
                        self._writer.add_scalar('loss', loss, training_step)

        for thread in threads:
            thread.join()


class Worker(Thread):

    def __init__(self, learner, buffer, training_episode, writer, worker_id=1):
        Thread.__init__(self, daemon=True)
        print(f'[{datetime.now().isoformat()}] Thread({threading.get_ident()})')
        self._env = gym.make(environment, no_graphics=True, worker_id=worker_id)
        self._model = LstmActorCriticModel(
            self._env.observation_space.shape[0] * sequence_length,
            self._env.action_space.n,
            hidden_size=256
        )
        self._worker_agent = AcerAgent(
            self._model,
            buffer,
            learning_rate=learning_rate,
            cuda=False
        )
        self._learner = learner
        self._training_episode = training_episode
        self._writer = writer
        """
        self._opponent_agent = PPOAgent(
            self._env.observation_space.shape[0] * sequence_length,
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

                obs1 = np.concatenate([new_obs1] * sequence_length)
                obs2 = np.concatenate([new_obs2] * sequence_length)

                rnn_output_size = self._worker_agent.rnn_output_size
                h_out = [(torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                          torch.zeros([1, 1, rnn_output_size], dtype=torch.float)),
                         (torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                          torch.zeros([1, 1, rnn_output_size], dtype=torch.float))]

                done = False

                while not done:
                    # for t in range(rollout):
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
                    loss, pi_loss, value_loss = self._learner._target_agent.apply_gradient(self._worker_agent, time_horizon=rollout)
                if not no_logging:
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
