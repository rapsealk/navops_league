#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
from itertools import count
from threading import Thread
from multiprocessing import cpu_count

import gym
import gym_rimpac   # noqa: F401
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import DQNAgent
from memory import ReplayBuffer
from utils import epsilon, SlackNotification
from rating import EloRating

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]


def discount_rewards(rewards: np.ndarray, gamma=0.998):  # 347 -> 0.5
    discounted = np.zeros_like(rewards)
    discounted[-1] = rewards[-1]
    n = len(rewards) - 1
    for i in range(n):
        discounted[n-i-1] = discounted[n-i] * gamma     # + rewards[n-i-1]
    return discounted


ENVIRONMENT = 'RimpacDiscrete-v0'
BATCH_SIZE = 2048
SEQUENCE = 64

HITPOINT_FIELD = -1
AMMO_FIELD = -4
FUEL_FIELD = -3


class Learner:

    def __init__(self):
        self._env = gym.make(ENVIRONMENT)
        self._target_agent = DQNAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            BATCH_SIZE
        )
        self._opponent_agent = DQNAgent(
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
        n = self._env.observation_space.shape[0]
        results = []
        ratings = (1200, 1200)
        best_rating = 1200
        for episode in count(1):
            new_obs1, new_obs2 = self._env.reset()

            obs1 = np.zeros((n * SEQUENCE,))
            obs1 = np.concatenate((obs1[n:], new_obs1))
            obs2 = np.zeros((n * SEQUENCE,))
            obs2 = np.concatenate((obs1[n:], new_obs2))

            while True:
                action1 = self._target_agent.get_action(obs1)
                action2 = self._opponent_agent.get_action(obs2)
                action = np.array([action1, action2], dtype=np.uint8)

                next_obs, reward, done, info = self._env.step(action)

                next_obs1 = np.concatenate((obs1[n:], next_obs[0]))
                next_obs2 = np.concatenate((obs2[n:], next_obs[1]))

                obs1, obs2 = next_obs1, next_obs2

                if done:
                    winner = info.get('win', 0)
                    if winner == -1:    # Draw
                        results.append(False)
                    else:
                        a_win = bool(1 - winner)
                        results.append(a_win)
                        ratings = EloRating.calc(ratings[0], ratings[1], a_win)
                    self._writer.add_scalar('r/rating', ratings[0], episode)

                    if ratings[0] > best_rating:
                        self._target_agent.save(
                            os.path.join(
                                os.path.dirname(__file__),
                                'checkpoints',
                                f'rimpac-discrete-v0-dqn-{episode:05}.ckpt'
                            )
                        )
                        best_rating = ratings[0]

                    if len(self._buffer) >= BATCH_SIZE:
                        loss = self._target_agent.train(self._buffer)
                        self._writer.add_scalar('loss/Q', loss, episode)

                    if episode % 100 == 0:
                        self._writer.add_scalar('r/result', np.mean(results), episode // 100)
                        results.clear()

                    if episode % 200 == 0:
                        self._opponent_agent.set_state_dict(self._target_agent.state_dict())
                        ratings = (ratings[0], ratings[0])

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
        self._agent1 = DQNAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            BATCH_SIZE,
            force_cpu=True
        )
        self._agent2 = DQNAgent(
            self._env.observation_space.shape[0] * SEQUENCE,
            self._env.action_space.n,
            BATCH_SIZE,
            force_cpu=True
        )

    def run(self):
        n = self._env.observation_space.shape[0]
        eps = epsilon()
        for episode in count(1):
            new_obs1, new_obs2 = self._env.reset()

            obs1 = np.zeros((n * SEQUENCE,))
            obs1 = np.concatenate((obs1[n:], new_obs1))
            obs2 = np.zeros((n * SEQUENCE,))
            obs2 = np.concatenate((obs1[n:], new_obs2))

            state_dict = self._target_agent.state_dict()
            self._agent1.set_state_dict(state_dict)
            self._agent2.set_state_dict(state_dict)

            buffer1, buffer2 = [], []

            while True:
                if next(eps) < np.random.random():
                    action1 = self._agent1.get_action(obs1)
                    action2 = self._agent2.get_action(obs2)
                else:
                    action1 = np.random.randint(self._env.action_space.n)
                    action2 = np.random.randint(self._env.action_space.n)
                action = np.array([action1, action2], dtype=np.uint8)

                next_obs, reward, done, info = self._env.step(action)

                next_obs1 = np.concatenate((obs1[n:], next_obs[0]))
                next_obs2 = np.concatenate((obs2[n:], next_obs[1]))

                buffer1.append((obs1, action1, reward[0], next_obs1, done))
                buffer2.append((obs2, action2, reward[1], next_obs2, done))

                obs1, obs2 = next_obs1, next_obs2

                if done:
                    trajectory1 = np.stack(buffer1)
                    trajectory2 = np.stack(buffer2)

                    winner = info.get('win', 0)
                    if winner == -1:    # Draw
                        hp_diff = trajectory1[-2, 3][HITPOINT_FIELD] - trajectory2[-2, 3][HITPOINT_FIELD]
                        if hp_diff > 0:
                            reward1 = hp_diff * np.mean((trajectory1[-2, 3][AMMO_FIELD], trajectory1[-2, 3][FUEL_FIELD]))  # Resource Loss
                            reward2 = 0
                        else:
                            reward2 = -hp_diff * np.mean((trajectory2[-2, 3][AMMO_FIELD], trajectory2[-2, 3][FUEL_FIELD]))
                            reward1 = 0
                    else:
                        reward1 = 1 - winner
                        reward2 = winner

                    trajectory1[:, 2] = reward1
                    trajectory2[:, 2] = reward2

                    trajectory1[:, 2] = discount_rewards(trajectory1[:, 2])
                    trajectory2[:, 2] = discount_rewards(trajectory2[:, 2])
                    for s, a, r, s_, d in np.concatenate([trajectory1, trajectory2]):
                        self._buffer.push(s, a, r, s_, d)

                    break


# @SlackNotification(SLACK_API_TOKEN)
def main():
    Learner().run()

    """
    env = gym.make('RimpacDiscrete-v0', mock=True)
    n = env.observation_space.shape[0]

    agent1 = DQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        BATCH_SIZE
    )
    agent2 = DQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        BATCH_SIZE
    )
    memory = ReplayBuffer()
    buffer1 = []
    buffer2 = []
    results = []
    ratings = (1200, 1200)
    best_rating = 1200

    writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-RimpacDiscrete-v0')

    eps = epsilon()
    for episode in count(1):
        new_obs1, new_obs2 = env.reset()

        obs1 = np.zeros((n * SEQUENCE,))
        obs1 = np.concatenate((obs1[n:], new_obs1))
        obs2 = np.zeros((n * SEQUENCE,))
        obs2 = np.concatenate((obs1[n:], new_obs2))

        while True:
            if next(eps) < np.random.random():
                action1 = agent1.get_action(obs1)
                action2 = agent2.get_action(obs2)
            else:
                action1 = np.random.randint(env.action_space.n)
                action2 = np.random.randint(env.action_space.n)
            action = np.array([action1, action2], dtype=np.uint8)

            next_obs, reward, done, info = env.step(action)

            next_obs1 = np.concatenate((obs1[n:], next_obs[0]))
            next_obs2 = np.concatenate((obs2[n:], next_obs[1]))

            buffer1.append((obs1, action1, reward[0], next_obs1, done))
            buffer2.append((obs2, action2, reward[1], next_obs2, done))

            obs1, obs2 = next_obs1, next_obs2

            if done:
                a_win = bool(1 - info.get('win', 0))
                results.append(a_win)
                ratings = EloRating.calc(ratings[0], ratings[1], a_win)
                writer.add_scalar('r/rating1', ratings[0], episode)
                writer.add_scalar('r/rating2', ratings[1], episode)

                trajectory1 = np.stack(buffer1)
                trajectory2 = np.stack(buffer2)
                trajectory1[:, 2] = discount_rewards(trajectory1[:, 2])
                trajectory2[:, 2] = discount_rewards(trajectory2[:, 2])
                for s, a, r, s_, d in np.concatenate([trajectory1, trajectory2]):
                    memory.push(s, a, r, s_, d)

                if len(memory) >= BATCH_SIZE:
                    loss1 = agent1.train(memory)
                    loss2 = agent2.train(memory)
                    writer.add_scalar('loss/q1', loss1, episode)
                    writer.add_scalar('loss/q2', loss2, episode)

                if episode % 100 == 0:
                    writer.add_scalar('r/result', np.mean(results), episode // 100)
                    results.clear()

                break
    """


if __name__ == "__main__":
    main()
