#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from collections import deque
from datetime import datetime
from itertools import count
from threading import Thread, Lock
from multiprocessing import cpu_count

import gym
import gym_rimpac   # noqa: F401
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import SoftActorCriticAgent, ReplayBuffer
from utils import epsilon
from rating import EloRating

parser = argparse.ArgumentParser()
parser.add_argument('--no-graphics', action='store_true', default=True)
parser.add_argument('--mock', action='store_true', default=False)
args = parser.parse_args()

ENVIRONMENT = 'Rimpac-v0'
# MOCK_ENVIRONMENT = 'Mock-Rimpac-v0'
BATCH_SIZE = 16
TIME_SEQUENCE = 4


def process_raw_observation(obs1, obs2, next_obs):
    next_obs = np.expand_dims(next_obs, axis=1)
    next_obs1 = np.concatenate((
        obs1[1:],
        np.expand_dims(np.concatenate((obs1[-1, 1:], next_obs[0])), axis=0)
    ))
    next_obs2 = np.concatenate((
        obs2[1:],
        np.expand_dims(np.concatenate((obs2[-1, 1:], next_obs[1])), axis=0)
    ))
    return next_obs1, next_obs2


class Learner:

    def __init__(self):
        env = gym.make(ENVIRONMENT, mock=True)
        self.global_agent = SoftActorCriticAgent(
            env.observation_space.shape[0],
            env.action_space,
            hidden_dim=64,
            num_layers=32,
            batch_size=BATCH_SIZE
        )
        env.close()
        del env

        self._buffer = ReplayBuffer(1_000_000)
        self._writer = SummaryWriter('runs/%s-%s' % (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), ENVIRONMENT))

        self._lock = Lock()
        self._num_workers = cpu_count() - 2
        self._value_step = 0
        self._policy_step = 0

    def run(self):
        workers = []
        for i in range(self._num_workers):
            workers.append(
                Worker(self, self.global_agent, i+1, self._buffer)
            )

        for worker in workers:
            worker.start()

        thread = Validator(self.global_agent, self._num_workers+1, self._writer)
        thread.start()

        for worker in workers:
            worker.join()

        thread.join()

    def update_parameters_by_worker_gradient(self, worker_model, q_loss=None, pi_loss=None, alpha_loss=None):
        with self._lock:
            if q_loss is not None:
                self._value_step += 1
                self.global_agent.descent_gradient(worker_model, q_loss=q_loss)
                self._writer.add_scalar('loss/value', q_loss, self._value_step)
            elif pi_loss is not None:
                self._policy_step += 1
                self.global_agent.descent_gradient(worker_model, pi_loss=pi_loss)
                self._writer.add_scalar('loss/policy', pi_loss, self._value_step)


class Worker(Thread):

    def __init__(
        self,
        learner,
        global_agent,
        worker_id=0,
        buffer=None
    ):
        Thread.__init__(self, daemon=True)
        self._worker_id = worker_id
        self._learner = learner
        self._buffer = buffer
        self._env = gym.make(
            ENVIRONMENT,
            worker_id=worker_id,
            no_graphics=args.no_graphics,
            mock=args.mock
        )
        self._observation_shape = self._env._observation_space.shape[0]

        self._global_agent = global_agent
        self._agent1 = SoftActorCriticAgent(
            self._observation_shape,
            self._env.action_space,
            hidden_dim=64,
            num_layers=32,
            batch_size=BATCH_SIZE,
            force_cpu=True
        )
        self._agent2 = SoftActorCriticAgent(
            self._observation_shape,
            self._env.action_space,
            hidden_dim=64,
            num_layers=32,
            batch_size=BATCH_SIZE,
            force_cpu=True
        )

        self.load_learner_parameters()

    def run(self):
        eps = epsilon()
        for episode in count(1):
            obs_batch1 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *self._env.observation_space.shape))  # 0.2 MB
            obs_batch2 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *self._env.observation_space.shape))
            memory1, memory2 = [], []

            obs = self._env.reset()
            obs_batch1, obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, obs)

            while True:
                if next(eps) > np.random.random():
                    action1 = np.random.uniform(-1.0, 1.0, (BATCH_SIZE, TIME_SEQUENCE, self._env.action_space.shape[0]))
                    action2 = np.random.uniform(-1.0, 1.0, (BATCH_SIZE, TIME_SEQUENCE, self._env.action_space.shape[0]))
                else:
                    action1 = self._agent1.select_action(obs_batch1)
                    action2 = self._agent2.select_action(obs_batch2)
                actual_action = np.concatenate((action1[-1, -1:], action2[-1, -1:]))

                next_obs, reward, done, info = self._env.step(actual_action)
                next_obs_batch1, next_obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, next_obs)

                memory1.append((obs_batch1[-1], action1[-1], reward[0], next_obs_batch1[-1], done))
                memory2.append((obs_batch2[-1], action2[-1], reward[1], next_obs_batch2[-1], done))

                obs_batch1, obs_batch2 = next_obs_batch1, next_obs_batch2

                if done:
                    memory1 = np.array(memory1)
                    memory1[:, 2] = 1 - info.get('win', 0)
                    memory2 = np.array(memory2)
                    memory2[:, 2] = info.get('win', 0)

                    for s, a, r, s_, d in np.concatenate((memory1, memory2)):
                        self._buffer.push(s, a, r, s_, d)
                    if len(self._buffer) >= BATCH_SIZE:
                        s, a, r, s_, dones = self._agent1.get_trajectory_batch(self._buffer, BATCH_SIZE)
                        q_loss = self._agent1.get_value_gradient(s, a, r, s_, dones)
                        self._learner.update_parameters_by_worker_gradient(self._agent1, q_loss=q_loss)
                        policy_loss = self._agent1.get_policy_gradient(s, a, r, s_, dones)
                        self._learner.update_parameters_by_worker_gradient(self._agent1, pi_loss=policy_loss)

                        s, a, r, s_, dones = self._agent2.get_trajectory_batch(self._buffer, BATCH_SIZE)
                        q_loss = self._agent2.get_value_gradient(s, a, r, s_, dones)
                        self._learner.update_parameters_by_worker_gradient(self._agent2, q_loss=q_loss)
                        policy_loss = self._agent2.get_policy_gradient(s, a, r, s_, dones)
                        self._learner.update_parameters_by_worker_gradient(self._agent2, pi_loss=policy_loss)

                    self.load_learner_parameters()
                    self._agent1.reset()
                    self._agent2.reset()

                    memory1, memory2 = [], []

        self._env.close()

    def load_learner_parameters(self):
        state_dict = self._global_agent.get_state_dict()
        self._agent1.set_state_dict(state_dict)
        self._agent2.set_state_dict(state_dict)


class Validator(Thread):

    def __init__(self, global_agent, worker_id=0, writer=None):
        Thread.__init__(self, daemon=True)
        self._worker_id = worker_id
        self._env = gym.make(
            ENVIRONMENT,
            worker_id=worker_id,
            no_graphics=True,
            mock=args.mock
        )
        self._observation_shape = self._env._observation_space.shape[0]

        self._writer = writer
        self._global_agent = global_agent
        self._target_agent = SoftActorCriticAgent(
            self._observation_shape,
            self._env.action_space,
            hidden_dim=64,
            num_layers=32,
            batch_size=BATCH_SIZE,
            force_cpu=True
        )
        self._opponent = SoftActorCriticAgent(
            self._observation_shape,
            self._env.action_space,
            hidden_dim=64,
            num_layers=32,
            batch_size=BATCH_SIZE,
            force_cpu=True
        )

        self.load_learner_parameters()

    def run(self):
        ratings = (1200, 1200)
        results = deque(maxlen=100)
        best_rate = 0
        for episode in count(1):
            obs_batch1 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *self._env.observation_space.shape))  # 0.2 MB
            obs_batch2 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *self._env.observation_space.shape))

            obs = self._env.reset()
            obs_batch1, obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, obs)

            while True:
                action1 = self._target_agent.select_action(obs_batch1)
                action2 = self._opponent.select_action(obs_batch2)
                actual_action = np.concatenate((action1[-1, -1:], action2[-1, -1:]))

                next_obs, reward, done, info = self._env.step(actual_action)
                obs_batch1, obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, next_obs)

                if done:
                    a_win = bool(1 - info['win'])
                    results.append(a_win)
                    ratings = EloRating.calc(ratings[0], ratings[1], a_win)
                    self._writer.add_scalar('r/rating', ratings[0], episode)
                    if episode % 100 == 0:
                        rate = np.mean(results)
                        if rate > best_rate:
                            self._target_agent.save(
                                os.path.join(
                                    os.path.dirname(__file__),
                                    'checkpoints',
                                    f'{ENVIRONMENT}-lstm-ep{episode}.ckpt')
                            )
                            best_rate = rate
                        self._writer.add_scalar('r/winnings', rate, episode // 100)
                        results.clear()
                    self.load_learner_parameters()
                    self._target_agent.reset()
                    break

        self._env.close()

    def load_learner_parameters(self):
        self._target_agent.set_state_dict(
            self._global_agent.get_state_dict()
        )


def main():
    learner = Learner()
    learner.run()


if __name__ == "__main__":
    main()
    """
    env = gym.make(ENVIRONMENT)
    agent = SoftActorCriticAgent(
        env.observation_space.shape[0],
        env.action_space,
        hidden_dim=64,
        num_layers=32,
        batch_size=BATCH_SIZE
    )
    buffer = ReplayBuffer(100000)

    for episode in count(1):
        obs_batch1 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *env.observation_space.shape))  # 0.2 MB
        obs_batch2 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *env.observation_space.shape))

        obs = env.reset()
        # print('obs.shape:', obs.shape)
        obs_batch1, obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, obs)
        # print('obs1.shape:', obs_batch1.shape)

        for _ in range(BATCH_SIZE):
            print('obs_batch:', obs_batch1.shape)
            action1 = agent.select_action(obs_batch1)
            # print('action1.shape:', action1.shape)
            action2 = agent.select_action(obs_batch2)
            # print('action2.shape:', action2.shape)

            # print('action1.shape:', action1.shape)
            # print('action1[-1, -1].shape:', action1[-1, -1:].shape)
            actual_action = np.concatenate((action1[-1, -1:], action2[-1, -1:]))
            print('action:', actual_action.shape)

            # print('actual_action:', actual_action.shape)
            next_obs, reward, done, _ = env.step(actual_action)
            # print('reward:', reward)
            # print('done:', done)

            next_obs_batch1, next_obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, next_obs)
            # print('next_obs1.shape:', obs_batch1.shape)

            buffer.push(obs_batch1[-1], action1[-1], reward[0], next_obs_batch1[-1], done)
            buffer.push(obs_batch2[-1], action2[-1], reward[1], next_obs_batch2[-1], done)


        # Train
        # q_loss, pi_loss, alpha_loss = agent.compute_gradient(buffer, BATCH_SIZE, episode)
        # print(f'q_loss: {q_loss}, pi_loss: {pi_loss}, alpha_loss: {alpha_loss}')
        # agent.descent_gradient(agent, q_loss, pi_loss, episode)
        s, a, r, s_, dones = agent.get_trajectory_batch(buffer, BATCH_SIZE)
        q_loss = agent.get_value_gradient(s, a, r, s_, dones)
        agent.descent_gradient(local=agent, q_loss=q_loss)
        policy_loss = agent.get_policy_gradient(s, a, r, s_, dones)
        agent.descent_gradient(local=agent, pi_loss=policy_loss)

        print(f'memory:', sys.getsizeof(buffer))

        env.close()
        break
    """
