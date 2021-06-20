#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import json
import importlib
from collections import deque
from datetime import datetime
from itertools import count
import threading
from threading import Thread, Lock
from multiprocessing import cpu_count

import gym
import gym_navops   # noqa: F401
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from memory import ReplayBuffer
from utils import generate_id   # SlackNotification, Atomic
from utils.board import ReportingBoard
from database import MongoDatabase


with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v0')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--worker-id', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--buffer-size', type=int, default=10000) # 42533 bytes -> 10000 (12GB)
parser.add_argument('--time-horizon', type=int, default=32)   # 2048
parser.add_argument('--seq-len', type=int, default=64)  # 0.1s per state-action
# parser.add_argument('--aggressive_factor', type=float, default=1.0)
# parser.add_argument('--defensive_factor', type=float, default=0.7)
parser.add_argument('--learning-rate', type=float, default=3e-5)
parser.add_argument('--no-logging', action='store_true', default=False)
parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], default='pytorch')
args = parser.parse_args()

if args.framework == 'tensorflow':
    models_impl = importlib.import_module('models.tensorflow_impl')
    # tf.summary.SummaryWriter
elif args.framework == 'pytorch':
    import torch
    models_impl = importlib.import_module('models.pytorch_impl')


# TODO: ML-Agents EventSideChannel(uuid.uuid4())

environment = args.env
# Hyperparameters
rollout = args.time_horizon
batch_size = args.batch_size
sequence_length = args.seq_len
# AGGRESSIVE_FACTOR = args.aggressive_factor
# DEFENSIVE_FACTOR = args.defensive_factor
learning_rate = args.learning_rate
no_logging = args.no_logging

workers = 0     # cpu_count()


class Learner:

    def __init__(self):
        self.session_id = generate_id()

        build_path = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOpsGrpc', 'NavOps.exe')
        print('build:', build_path)
        self._env = gym.make(environment, build_path=build_path)
        self._buffer = ReplayBuffer(args.buffer_size)
        self._target_model = models_impl.MultiHeadLstmActorCriticModel(
            self._env.observation_space.shape[0] * sequence_length,
            self._env.action_space.nvec,
            hidden_size=512
        )
        self._target_agent = models_impl.MultiHeadAcerAgent(
            model=self._target_model,
            buffer=self._buffer,
            learning_rate=learning_rate,
            cuda=True
        )
        self._id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{environment}'
        if not no_logging:
            self._writer = SummaryWriter(f'runs/{self._id}')
            self._plotly = ReportingBoard()
            database = MongoDatabase()
            self._result_db = database.ref("result")
            self._session_db = database.ref(self.session_id)

        if not args.no_logging:
            with open(os.path.join(os.path.dirname(__file__), f'{self._id}.log'), 'w') as f:
                experiment_settings = {
                    "session": self.session_id,
                    "id": self._id,
                    "framework": args.framework,
                    "environment": environment,
                    "time_horizon": args.time_horizon,
                    "batch_size": args.batch_size,
                    "sequence_length": args.seq_len,
                    "learning_rate": args.learning_rate
                }
                f.write(json.dumps(experiment_settings))

        # self._training_episode = Atomic(int)
        self._lock = Lock()

    def run(self):
        threads = [
            Worker(self, self._buffer, self._training_episode, self._writer, args.worker_id+i+1)
            for i in range(workers)
        ]
        for thread in threads:
            thread.start()

        observation_shape = self._env.observation_space.shape[0]

        result_wins_dq = deque(maxlen=10)
        result_draws_dq = deque(maxlen=10)
        result_loses_dq = deque(maxlen=10)
        result_episodes_dq = deque(maxlen=10)
        result_wins = []
        result_draws = []
        result_loses = []

        training_step = 0
        for episode in count(1):
            rewards = []
            observations = self._env.reset()

            if args.framework == 'pytorch':
                h_out = self._target_agent.reset_hidden_state(batch_size=1)

            done = False

            episode_id = ('0' * 10 + str(episode))[-10:]
            ref = self._session_db.ref(episode_id)

            while not done:
                batch = []
                for t in range(rollout):
                    if args.framework == 'tensorflow':
                        (action1_m, prob1_m), (action1_a, prob1_a) = self._target_agent.get_action(observations[0])
                    elif args.framework == 'pytorch':
                        h_in = h_out.copy()
                        (action1_m, prob1_m), (action1_a, prob1_a), h_out[0] = self._target_agent.get_action(observations[0], h_in[0])
                    action = np.array([[[action1_m, action1_a]] * args.n]).squeeze()

                    next_observations, reward, done, info = self._env.step(action)

                    rewards.append(reward)

                    if args.framework == 'tensorflow':
                        batch.append((observations[0], action, reward, next_observations[0], (prob1_m, prob1_a), not done))
                    elif args.framework == 'pytorch':
                        batch.append((observations[0], action, reward, next_observations[0], (prob1_m, prob1_a), h_in[0], h_out[0], not done))

                    value = {
                        "hp": info['obs'].fleets[0].hp,
                        "position": [
                            info['obs'].fleets[0].position.x,
                            info['obs'].fleets[0].position.y
                        ],
                        "rotation": [
                            info['obs'].fleets[0].rotation.cos,
                            info['obs'].fleets[0].rotation.sin
                        ],
                        "opponent": {
                            "hp": info['obs'].fleets[1].hp,
                            "position": [
                                info['obs'].fleets[1].position.x,
                                info['obs'].fleets[1].position.y
                            ],
                            "rotation": [
                                info['obs'].fleets[1].rotation.cos,
                                info['obs'].fleets[1].rotation.sin
                            ]
                        },
                        "action": action,
                        "reward": reward
                    }
                    # print(f'[main] value: {value}')
                    _ = ref.put(**value)

                    if done:
                        print(f'[{datetime.now().isoformat()}] Episode #{episode} Done! -> {info.get("win", False)}')

                        result_wins.append(info.get('win', False) is True)
                        result_loses.append(info.get('win', False) is False)
                        # result_draws.append(info.get('win', False) is False)
                        result_draws.append(False)

                        if not no_logging:
                            self._writer.add_scalar('r/rewards', np.sum(rewards), episode)
                            self._writer.add_scalar('logging/hitpoint', info['obs'].fleets[0].hp, episode)
                            # self._writer.add_scalar('logging/hitpoint_gap', obs1[field_hitpoint] - obs2[field_hitpoint], episode)
                            # self._writer.add_scalar('logging/damage', 1 - obs2[field_hitpoint], episode)
                            self._writer.add_scalar('logging/ammo_usage', 1 - info['obs'].ammo, episode)
                            self._writer.add_scalar('logging/fuel_usage', 1 - info['obs'].fleets[0].fuel, episode)
                            if episode % 100 == 0:
                                result_wins_dq.append(np.sum(result_wins))
                                result_draws_dq.append(np.sum(result_draws))
                                result_loses_dq.append(np.sum(result_loses))
                                result_episodes_dq.append(str(episode))
                                result_wins = []
                                result_draws = []
                                result_loses = []
                                # data = [tuple(result_wins_dq), tuple(result_draws_dq), tuple(result_loses_dq)]
                                #self._plotly.plot(tuple(result_episodes_dq), data)
                                #self._plotly.plot_scatter(data)
                                wins = tuple(result_wins_dq)
                                draws = tuple(result_draws_dq)
                                loses = tuple(result_loses_dq)
                                self._plotly.plot_winning_rate(wins, draws, loses)
                                # self._writer.add_scalar('r/wins', np.mean(result_wins), episode)
                                # self._writer.add_scalar('r/loses', np.mean(result_loses), episode)
                                # self._writer.add_scalar('r/draws', np.mean(result_draws), episode)

                                self._target_agent.save(os.path.join(os.path.dirname(__file__), 'checkpoints', f'{environment}-acer-{episode}.ckpt'), episode=episode)
                        break

                    # obs1, obs2 = next_obs1, next_obs2
                    observations = next_observations

                self._buffer.push(batch)
                if len(self._buffer) > 5:#00:
                    training_step += 1
                    loss = self._target_agent.train(batch_size, on_policy=True)
                    loss += self._target_agent.train(batch_size)
                    print(f'[{datetime.now().isoformat()}] Loss: {loss} (batch: {len(self._buffer)})')
                    if not no_logging:
                        self._writer.add_scalar('loss', loss, training_step)

                    if done and not no_logging:
                        self._result_db.put(**{
                            "session": self.session_id,
                            "episode": episode_id,
                            "result": info.get('win', False),
                            "performance": {
                                "hp": info['obs'].fleets[0].hp,
                                "fuel": info['obs'].fleets[0].fuel,
                                "ammo": info['obs'].ammo
                            },
                            "reward": np.sum(rewards),
                            "loss": loss
                        })

        for thread in threads:
            thread.join()


class Worker(Thread):

    def __init__(self, learner, buffer, training_episode, writer, worker_id=1):
        Thread.__init__(self, daemon=True)
        print(f'[{datetime.now().isoformat()}] Thread({threading.get_ident()})')
        self._env = gym.make(environment, no_graphics=True, worker_id=worker_id)
        self._model = models_impl.MultiHeadLstmActorCriticModel(
            self._env.observation_space.shape[0] * sequence_length,
            self._env.action_space.nvec,
            hidden_size=256
        )
        self._worker_agent = models_impl.MultiHeadAcerAgent(
            self._model,
            buffer,
            learning_rate=learning_rate,
            cuda=False
        )
        self._buffer = buffer
        self._learner = learner
        self._training_episode = training_episode
        self._writer = writer

    def run(self):
        observation_shape = self._env.observation_space.shape[0]
        while True:
            rewards = []
            new_obs1, new_obs2 = self._env.reset()

            obs1 = np.concatenate([new_obs1] * sequence_length)
            obs2 = np.concatenate([new_obs2] * sequence_length)

            rnn_output_size = self._worker_agent.rnn_output_size
            h_out = [(torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float)),
                     (torch.zeros([1, 1, rnn_output_size], dtype=torch.float),
                      torch.zeros([1, 1, rnn_output_size], dtype=torch.float))]

            done = False

            self.load_learner_parameters()

            while not done:
                batch = []
                for t in range(rollout):
                    h_in = h_out.copy()
                    action1, prob1, h_out[0] = self._worker_agent.get_action(obs1, h_in[0])
                    # action2, prob2, h_out[1] = self._opponent_agent.get_action(obs2, h_in[1])
                    action2 = np.random.randint(self._env.action_space.nvec)
                    action = np.array([action1, action2], dtype=np.uint8)

                    next_obs, reward, done, info = self._env.step(action)

                    rewards.append(reward)

                    next_obs1 = np.concatenate((obs1[observation_shape:], next_obs[0]))
                    next_obs2 = np.concatenate((obs2[observation_shape:], next_obs[1]))

                    batch.append((obs1, action[0], reward, next_obs1, prob1, h_in[0], h_out[0], not done))

                    if done:
                        break

                    obs1, obs2 = next_obs1, next_obs2

                self._buffer.push(batch)

    def load_learner_parameters(self):
        self._worker_agent.set_state_dict(
            self._learner._target_agent.state_dict()
        )


# @SlackNotification(SLACK_API_TOKEN)
def main():
    Learner().run()


if __name__ == "__main__":
    main()
