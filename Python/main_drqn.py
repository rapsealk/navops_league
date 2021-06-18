#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import json
import time
from collections import deque
from datetime import datetime
from itertools import count
from threading import Lock
# from multiprocessing import cpu_count

import gym
import gym_navops   # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import generate_id   # SlackNotification, Atomic
from utils.board import ReportingBoard
from utils.database import MongoDatabase
from drqn_utils import EpisodeBuffer, EpisodeMemory
from models.pytorch_impl import DRQNAgent


with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v0')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--buffer-size', type=int, default=20000)
parser.add_argument('--time-horizon', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=128)
parser.add_argument('--learning-rate', type=float, default=1e-3)    # 3e-5
parser.add_argument('--target-update-period', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--tau', type=float, default=1e-2)
parser.add_argument('--no-logging', action='store_true', default=False)
args = parser.parse_args()


def discount_rewards(rewards, gamma=0.98):
    rewards_ = np.zeros_like(rewards)
    rewards_[0] = rewards[-1]
    for i in range(1, len(rewards)):
        rewards_[i] = rewards_[i-1] * gamma + rewards[-i-1]
    return rewards_[::-1]


class Learner:

    def __init__(self):
        self.session_id = generate_id()

        build_path = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOps', 'NavOps.exe')
        print('build:', build_path)
        self._env = gym.make(args.env, build_path=build_path)
        self._agent = DRQNAgent(
            input_size=self._env.observation_space.shape[0],
            action_sizes=self._env.action_space.nvec,
            hidden_size=512,
            rnn_hidden_size=64,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_rate=args.learning_rate)

        self._memory = EpisodeMemory(max_epi_num=args.buffer_size,
                                     max_epi_len=args.time_horizon,
                                     batch_size=args.batch_size,
                                     lookup_step=args.sequence_length,
                                     random_update=True)
        self._epsilon = 1.0
        self._epsilon_decay = 0.99
        self._epsilon_min = 0.01

        self._id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{args.env}'
        if not args.no_logging:
            self._writer = SummaryWriter(f'runs/{self._id}')
            self._plotly = ReportingBoard()
            database = MongoDatabase()
            self._result_db = database.ref("result")
            self._session_db = database.ref(self.session_id)
            self._loss_db = self._session_db.ref("loss")

        if not args.no_logging:
            with open(os.path.join(os.path.dirname(__file__), f'{self._id}.log'), 'w') as f:
                experiment_settings = {
                    "session": self.session_id,
                    "id": self._id,
                    "environment": args.env,
                    "model": "Deep Recurrent Q Network",
                    "time_horizon": args.time_horizon,
                    "batch_size": args.batch_size,
                    "buffer_size": args.buffer_size,
                    "sequence_length": args.sequence_length,
                    "learning_rate": args.learning_rate,
                    "target_update_period": args.target_update_period,
                    "gamma": args.gamma,
                    "tau": args.tau
                }
                f.write(json.dumps(experiment_settings))

    def run(self):
        observation_shape = self._env.observation_space.shape[0]

        result_wins_dq = deque(maxlen=10)
        result_draws_dq = deque(maxlen=10)
        result_loses_dq = deque(maxlen=10)
        result_episodes_dq = deque(maxlen=10)
        result_wins = []
        result_draws = []
        result_loses = []

        train_step = 0

        time_ = time.time()

        for episode in count(1):
            state = self._env.reset()
            h_in = self._agent.reset_hidden_state(batch_size=1)

            episode_buffer = EpisodeBuffer()
            rewards = []

            done = False

            if not args.no_logging:
                episode_id = ('0' * 10 + str(episode))[-10:]
                ref = self._session_db.ref(episode_id)

            while not done:
                actions, _, h_in = self._agent.get_actions(state, h_in, epsilon=self._epsilon)

                next_state, reward, done, info = self._env.step(actions)
                episode_buffer.put(state, actions, reward, next_state, done)

                # print(f'Ep:{episode}:{t} state: {state}, action: {action} reward: {reward}, next_state: {next_state}, done: {done}')
                # print(f'Ep:{episode}:{t} state: {state}')

                if not args.no_logging:
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
                        "action": actions,
                        "reward": reward
                    }
                    # print(f'[main] value: {value}')
                    _ = ref.put(**value)

                state = next_state
                rewards.append(reward)

                if len(self._memory) >= args.batch_size:
                    loss = self._agent.train(episode_memory=self._memory,
                                             learning_rate=args.learning_rate,
                                             gamma=args.gamma,
                                             batch_size=args.batch_size)
                    if not args.no_logging:
                        self._writer.add_scalar('loss', loss, train_step)
                        self._loss_db.put({
                            "step": train_step,
                            "loss": loss
                        })
                    if (train_step := train_step + 1) % args.target_update_period == 0:
                        self._agent.hard_update()

                if done:
                    result = info.get('win', False)
                    result_wins.append(result is True)
                    result_loses.append(result is False)
                    result_draws.append(False)
                    result_sign = 'W' if result else 'L'

                    t_ = time.time() - time_
                    print(f'[{datetime.now().isoformat()}] ({int(t_//3600):02d}h {int(t_%3600//60):02d}m {(t_%3600)%60:02.2f}s) Ep:{episode} -> {result_sign}({np.sum(rewards)}) (epsilon={self._epsilon})')
                    self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)
                    episode_buffer.reward = discount_rewards(episode_buffer.reward)
                    self._memory.put(episode_buffer)

                    if not args.no_logging:
                        self._writer.add_scalar('r/rewards', np.sum(rewards), episode)
                        self._writer.add_scalar('logging/hitpoint', info['obs'].fleets[0].hp, episode)
                        # self._writer.add_scalar('logging/hitpoint_gap', obs1[field_hitpoint] - obs2[field_hitpoint], episode)
                        # self._writer.add_scalar('logging/damage', 1 - obs2[field_hitpoint], episode)
                        self._writer.add_scalar('logging/ammo_usage', 1 - info['obs'].ammo, episode)
                        self._writer.add_scalar('logging/fuel_usage', 1 - info['obs'].fleets[0].fuel, episode)

                        self._result_db.put(**{
                            "session": self.session_id,
                            "episode": episode_id,
                            "result": info.get('win', False),
                            "performance": {
                                "hp": info['obs'].fleets[0].hp,
                                "fuel": info['obs'].fleets[0].fuel,
                                "ammo": info['obs'].ammo
                            },
                            "reward": np.sum(rewards)
                        })

                        if episode % 100 == 0:
                            result_wins_dq.append(np.sum(result_wins))
                            result_draws_dq.append(np.sum(result_draws))
                            result_loses_dq.append(np.sum(result_loses))
                            result_episodes_dq.append(str(episode))
                            result_wins = []
                            result_draws = []
                            result_loses = []
                            wins = tuple(result_wins_dq)
                            draws = tuple(result_draws_dq)
                            loses = tuple(result_loses_dq)
                            self._plotly.plot_winning_rate(wins, draws, loses)
                            print(f'[{datetime.now().isoformat()}] Rate: {np.sum(wins)}% ({episode} Episodes)')

                            self._target_agent.save(os.path.join(os.path.dirname(__file__), 'checkpoints', f'{args.env}-drqn-{episode}.ckpt'), episode=episode)
                    break

        self._env.close()


# @SlackNotification(SLACK_API_TOKEN)
def main():
    Learner().run()


if __name__ == "__main__":
    main()
