#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
from collections import deque
from datetime import datetime
from threading import Thread, Lock

import gym
import gym_rimpac   # noqa: F401
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import cpu_count, Queue
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import SoftActorCriticAgent
from memory import MongoReplayBuffer as ReplayBuffer
# from memory import MongoLocalMemory as LocalMemory
from utils import SlackNotification
from rating import EloRating

os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--no-graphics', action='store_true', default=True)
parser.add_argument('--mock', action='store_true', default=False)
args = parser.parse_args()

ENVIRONMENT = 'Rimpac-v0'
BATCH_SIZE = 32
TIME_SEQUENCE = 4
HIDDEN_SIZE = 256
LAYERS = 32

EPSILON_DISCOUNT = 1e-3
EPSILON_STEP = 100
EPSILON_MIN = 5e-3


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
            env.action_space.shape[0],
            hidden_size=HIDDEN_SIZE,
            action_space=env.action_space,
            num_layers=LAYERS,
            batch_size=BATCH_SIZE,
            force_cpu=(sys.platform=='win32')   # Windows
        )
        env.close()
        del env

        self.global_agent.share_memory()

        self._buffer = ReplayBuffer()
        self._queue = Queue()
        self._writer = SummaryWriter('runs/%s-%s' % (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), ENVIRONMENT))

        self._lock = Lock()
        self._num_workers = 1   # cpu_count()
        self._step = 0

    def run(self):
        workers = [
            Worker(self, 0, self._queue, True)
        ]
        for i in range(self._num_workers):
            workers.append(
                Worker(self, i+1, self._queue)
            )

        for worker in workers:
            worker.start()

        queue_thread = Thread(target=self._digest_queue, args=(self._queue, self._buffer), daemon=True)
        queue_thread.start()

        for worker in workers:
            worker.join()

        queue_thread.join()

    def descent(self, gradient, q_loss, pi_loss, worker=None):
        """FIXME: mp.Queue"""
        with self._lock:
            self._step += 1
            self.global_agent.descent(gradient, worker)
            self._writer.add_scalar('loss/value', q_loss, self._step)
            self._writer.add_scalar('loss/policy', pi_loss, self._step)

    def update_rating_on_tensorboard(self, rating, step):
        self._writer.add_scalar('r/rating', rating, step)

    def update_winning_rate_on_tensorboard(self, rate, step):
        self._writer.add_scalar('r/winnings', rate, step)

    def update_time_on_tensorboard(self, seconds, step):
        self._writer.add_scalar('r/time', seconds, step)

    def update_random_action_rate_on_tensorboard(self, random_rate, step):
        self._writer.add_scalar('r/random_action', random_rate, step)

    def _digest_queue(self, queue, buffer):
        while True:
            s, a, r, s_, d = queue.get()
            buffer.push(s, a, r, s_, d)


class Worker(mp.Process):

    def __init__(
        self,
        learner,
        worker_id=0,
        queue=None,
        evaluate=False
    ):
        super(Worker, self).__init__(daemon=True)
        self._worker_id = worker_id
        self._learner = learner
        self._queue = queue
        self._evaluate = evaluate
        self._env = gym.make(
            ENVIRONMENT,
            worker_id=worker_id,
            no_graphics=args.no_graphics,
            mock=args.mock
        )
        self._observation_shape = self._env._observation_space.shape[0]

        self._global_agent = learner.global_agent
        self._agent1 = SoftActorCriticAgent(
            self._observation_shape,
            self._env.action_space.shape[0],
            hidden_size=HIDDEN_SIZE,
            action_space=self._env.action_space,
            num_layers=LAYERS,
            batch_size=BATCH_SIZE,
            optim_=self._global_agent.optim,
            force_cpu=True
        )
        self._agent2 = SoftActorCriticAgent(
            self._observation_shape,
            self._env.action_space.shape[0],
            hidden_size=HIDDEN_SIZE,
            action_space=self._env.action_space,
            num_layers=LAYERS,
            batch_size=BATCH_SIZE,
            optim_=self._global_agent.optim,
            force_cpu=True
        )

        self.load_learner_parameters()

    def run(self):
        ratings = (1200, 1200)
        results = deque(maxlen=100)
        best_rate = 0
        # eps = epsilon()
        eps = 1.0
        total_timestep = 0
        episode = 0
        # for episode in count(1):
        while True:
            episode += 1
            obs_batch1 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *self._env.observation_space.shape))  # 0.2 MB
            obs_batch2 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *self._env.observation_space.shape))
            memory1, memory2 = [], []

            obs = self._env.reset()
            obs_batch1, obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, obs)

            time_ = time.time()
            random_action = 0
            timestep = 0
            # for timestep in count(1):
            while True:
                timestep += 1
                total_timestep += 1
                if total_timestep % EPSILON_STEP == 0:
                    eps = max(eps - EPSILON_DISCOUNT, EPSILON_MIN)
                    total_timestep = 0
                # if next(eps) > np.random.random():
                if eps > np.random.random():
                    random_action += 1
                    action1 = np.random.uniform(-1.0, 1.0, (BATCH_SIZE, TIME_SEQUENCE, self._env.action_space.shape[0]))
                    action2 = np.random.uniform(-1.0, 1.0, (BATCH_SIZE, TIME_SEQUENCE, self._env.action_space.shape[0]))
                else:
                    action1 = self._agent1.get_action(obs_batch1, evaluate=self._evaluate)
                    action2 = self._agent2.get_action(obs_batch2, evaluate=self._evaluate)
                actual_action = np.concatenate((action1[-1, -1:], action2[-1, -1:]))

                next_obs, reward, done, info = self._env.step(actual_action)
                next_obs_batch1, next_obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, next_obs)

                memory1.append((obs_batch1[-1], action1[-1], reward[0], next_obs_batch1[-1], done))
                memory2.append((obs_batch2[-1], action2[-1], reward[1], next_obs_batch2[-1], done))
                # self._queue.put((obs_batch1[-1], action1[-1], reward[0], next_obs_batch1[-1], done))
                # self._queue.put((obs_batch2[-1], action2[-1], reward[1], next_obs_batch2[-1], done))

                obs_batch1, obs_batch2 = next_obs_batch1, next_obs_batch2

                if done:
                    if self._evaluate:
                        a_win = bool(1 - info.get('win', 0))
                        results.append(a_win)
                        ratings = EloRating.calc(ratings[0], ratings[1], a_win)
                        self._learner.update_rating_on_tensorboard(ratings[0], episode)
                        self._learner.update_time_on_tensorboard(time.time() - time_, episode)
                        self._learner.update_random_action_rate_on_tensorboard(random_action / timestep, episode)
                        if episode % 100 == 0:
                            rate = np.mean(results)
                            if rate > best_rate:
                                self._agent1.save(
                                    os.path.join(
                                        os.path.dirname(__file__),
                                        'checkpoints',
                                        f'{ENVIRONMENT}-lstm-ep{episode}.ckpt')
                                )
                                best_rate = rate
                            self._learner.update_winning_rate_on_tensorboard(rate, episode // 100)
                            results.clear()
                    else:
                        # Reward shaping
                        m1 = np.array(memory1)
                        m1[:, 2] = 1 - info.get('win', 0)
                        m2 = np.array(memory2)
                        m2[:, 2] = info.get('win', 0)

                        memory1.clear()
                        memory2.clear()

                        for s, a, r, s_, d in np.concatenate((m1, m2)):
                            # self._buffer.push(s, a, r, s_, d)
                            self._queue.put((s, a, r, s_, d))
                        # if len(self._buffer) >= BATCH_SIZE:
                        gradient, q_loss, pi_loss = self._agent1.compute_gradient(self._buffer, BATCH_SIZE)
                        self._learner.descent(gradient, q_loss, pi_loss, self._agent1)
                        gradient, q_loss, pi_loss = self._agent2.compute_gradient(self._buffer, BATCH_SIZE)
                        self._learner.descent(gradient, q_loss, pi_loss, self._agent2)

                    self.load_learner_parameters()
                    self._agent1.reset()
                    self._agent2.reset()

                    break

        self._env.close()

    def load_learner_parameters(self):
        state_dict = self._global_agent.get_state_dict()
        self._agent1.set_state_dict(state_dict)
        if not self._evaluate:
            self._agent2.set_state_dict(state_dict)


@SlackNotification
def main():
    learner = Learner()
    learner.run()


if __name__ == "__main__":
    main()
    # env = gym.make('Rimpac-v0', mock=True)
    # obs = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *env.observation_space.shape))
    # print(sys.getsizeof(obs) / 1000)
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
