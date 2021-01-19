#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import time
# import queue
import random
from datetime import datetime
from itertools import count
from threading import Thread, Lock, get_ident
from multiprocessing import cpu_count   # , Queue

import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import SoftActorCriticAgent, ReplayBuffer
# from models.pytorch_impl import MPReplayBuffer as ReplayBuffer


def epsilon():
    eps = 1.0
    eps_discount = 0.001
    eps_discount_step = 1000
    eps_minimum = 0.005
    for i in count(1):
        if i % eps_discount_step == 0:
            eps = max(eps - eps_discount, eps_minimum)
        yield eps


class Learner:

    def __init__(self):
        env = gym.make('MountainCarContinuous-v0')
        self.global_agent = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
        env.close()
        del env

        self._buffer = ReplayBuffer(1000000)
        # print('[{}] Buffer.id: {}'.format(datetime.now().isoformat(), id(self._buffer)))
        self._writer = SummaryWriter('runs/{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        self.num_workers = cpu_count()

    def run(self):
        workers = []
        writers = [self._writer] + [None] * (self.num_workers - 1)
        for i in range(self.num_workers):
            workers.append(
                Worker(self.global_agent, self._buffer, writers[i]),
            )

        thread = Thread(target=self.train, args=(self._buffer,), daemon=True)
        thread.start()

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        thread.join()

    def train(self, buffer):
        t = 1
        batch_size = 2048
        while True:
            # time.sleep(10)
            # print('[{}] len(_buffer): {}'.format(datetime.now().isoformat(), len(buffer)))
            if len(buffer) < batch_size:
                time.sleep(1)
                continue
            """
            if len(self._buffer) < 128:
                time.sleep(1)
                continue
            """

            q1, q2, pi, alpha, alpha_tlog = self.global_agent.update_parameters(buffer, batch_size, t)

            # SummaryWriter
            self._writer.add_scalar('loss/b/critic_1', q1, t)
            self._writer.add_scalar('loss/b/critic_2', q2, t)
            self._writer.add_scalar('loss/b/policy', pi, t)
            self._writer.add_scalar('loss/b/entropy_loss', alpha, t)
            self._writer.add_scalar('entropy_temprature/b/alpha', alpha_tlog, t)

            if t % 100 == 0:
                print('[{}] Episode {}: Q({}, {}), PI({})'.format(datetime.now().isoformat(), t, q1, q2, pi))
            t += 1


class Worker(Thread):

    def __init__(self, global_agent, buffer=None, writer=None):
        Thread.__init__(self, daemon=True)

        self._buffer = buffer   # or ReplayBuffer(1000000)
        # self._lock = Lock()
        self._env = gym.make('MountainCarContinuous-v0')

        self.global_agent = global_agent
        # self.max_episodes = 0
        self.agent = SoftActorCriticAgent(self._env.observation_space.shape[0], self._env.action_space)
        # self.gamma = gamma
        self.writer = writer

        self.update()

    def run(self):
        eps = epsilon()
        for episode in count(1):
            observation = self._env.reset()
            total_rewards = 0

            while True:
                if random.random() > next(eps):
                    action = self.agent.select_action(observation)
                else:
                    action = np.random.uniform(self._env.action_space.low[0],
                                               self._env.action_space.high[0],
                                               self._env.action_space.shape)
                    # action = np.random.random(self._env.action_space.shape[0])

                next_observation, reward, done, info = self._env.step(action)

                if self.writer is not None:
                    self._env.render()

                # with self._lock:
                self._buffer.push(observation, action, reward, next_observation, done)
                # print('[{}] tid: {} - {}({})'.format(datetime.now().isoformat(), get_ident(), len(self._buffer), id(self._buffer)))

                observation = next_observation
                total_rewards += reward

                if done:
                    if self.writer is not None:
                        self.writer.add_scalar('reward', total_rewards, episode)
                        print('[{}] Reward Ep. {}: {}'.format(datetime.now().isoformat(), episode, total_rewards))
                    self.update()
                    break

        self._env.close()

    def update(self):
        self.agent.set_state_dict(self.global_agent.get_state_dict())


def main():
    agent = Learner()
    agent.run()


if __name__ == "__main__":
    main()
