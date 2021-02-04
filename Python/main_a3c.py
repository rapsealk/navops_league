#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from datetime import datetime
from itertools import count
from threading import Thread, Lock
from multiprocessing import cpu_count

import numpy as np
import gym
import gym_rimpac   # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import SoftActorCriticAgent, ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--no-graphics', action='store_true', default=True)
args = parser.parse_args()

ENVIRONMENT = 'Rimpac-v0'
DISCOUNT_FACTOR = 0.998
BATCH_SIZE = 2048


def discount_rewards(rewards, dones, gamma=DISCOUNT_FACTOR):
    rewards = np.array(rewards, dtype=np.float32)
    returns = np.append(np.zeros_like(rewards), np.zeros((1, 1))).reshape((-1, 1))
    for t in reversed(range(rewards.shape[0])):
        returns[t] = rewards[t] + gamma * returns[t+1] * (1 - dones[t])
    return returns[:-1]


def epsilon():
    eps = 1.0
    eps_discount = 0.001
    eps_discount_step = 100
    eps_minimum = 0.005
    for i in count(1):
        if i % eps_discount_step == 0:
            eps = max(eps - eps_discount, eps_minimum)
        yield eps


def reevaluate_ratings(r_a: int, r_b: int, a_wins: bool, p=400):
    q_a = pow(10, r_a / p)
    q_b = pow(10, r_b / p)
    prob_a = q_a / (q_a + q_b)
    prob_b = q_b / (q_a + q_b)
    b_wins = 1 - a_wins
    r_a_ = r_a + round(k(r_a) * (a_wins - prob_a))
    r_b_ = r_b + round(k(r_b) * (b_wins - prob_b))
    return r_a_, r_b_


def k(rating: int):
    if rating > 2400:
        return 16
    elif rating > 2100:
        return 24
    return 32


class Learner:

    def __init__(self):
        env = gym.make(ENVIRONMENT, no_graphics=args.no_graphics)
        self.global_agent = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
        self._opponent = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
        env.close()
        del env

        self._buffer = ReplayBuffer(1000000)
        self._writer = SummaryWriter('runs/{}-{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), ENVIRONMENT))
        print('Rimpac:', 'runs/{}-{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), ENVIRONMENT))

        self._lock = Lock()
        self.num_workers = cpu_count()
        self._step = 0

    def run(self):
        workers = []
        writers = (self._writer,) + (None,) * self.num_workers
        for i in range(self.num_workers):
            workers.append(
                Worker(self.global_agent, i+1, self._buffer, writers[i]),
            )

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        self._env.close()

    def update_parameters_by_worker_gradient(self, worker_model, q_loss, pi_loss, alpha_loss):
        with self._lock:
            self._step += 1
            self.global_agent.descent_gradient(worker_model, q_loss, pi_loss, self._step)
            self._writer.add_scalar('Q Loss', q_loss, self._step)
            self._writer.add_scalar('Policy Loss', pi_loss, self._step)
            self._writer.add_scalar('Alpha Loss', alpha_loss, self._step)


class Worker(Thread):

    def __init__(self, global_agent, worker_id=0, buffer=None, writer=None):
        Thread.__init__(self, daemon=True)

        self._worker_id = worker_id
        self._buffer = buffer
        self._writer = writer
        self._env = gym.make(ENVIRONMENT, worker_id=worker_id, no_graphics=args.no_graphics)

        self.global_agent = global_agent
        self.agent1 = SoftActorCriticAgent(self._env.observation_space.shape[0], self._env.action_space)
        self.agent2 = SoftActorCriticAgent(self._env.observation_space.shape[0], self._env.action_space)
        # self.gamma = gamma

        self.load_learner_parameters()

    def run(self):
        ratings = (1200, 1200)
        best_score = 0
        eps = epsilon()
        for episode in count(1):
            observation = self._env.reset()
            memory1, memory2 = [], []

            while True:
                if np.random.random() > next(eps):
                    actions = (self.agent1.select_action(observation[0])[0],
                               self.agent2.select_action(observation[1])[0])
                else:
                    actions = (np.random.uniform(0.0, 1.0, size=(2,)+self._env.action_space.shape))
                    actions[:, :2] = (actions[:, :2] - 0.5) * 2
                    actions[:, 8:] = (actions[:, 8:] - 0.5) * 2
                action = np.stack(actions, axis=0)  # .squeeze(0)
                next_observation, reward, done, info = self._env.step(action)

                memory1.append((observation[0, 0], actions[0], reward[0], next_observation[0, 0], done))
                memory2.append((observation[1, 0], actions[1], reward[1], next_observation[1, 0], done))

                observation = next_observation

                if done:
                    mem1 = np.array(memory1)
                    mem1[:, 2] = discount_rewards(mem1[:, 2], mem1[:, 4]).squeeze()
                    mem2 = np.array(memory2)
                    mem2[:, 2] = discount_rewards(mem2[:, 2], mem2[:, 4]).squeeze()
                    for s, a, r, s_, d in np.concatenate([mem1, mem2]):
                        self._buffer.push(s, a, r, s_, d)
                    if len(self._buffer) >= BATCH_SIZE:
                        qf_loss, policy_loss, alpha_loss = self.agent1.compute_gradient(self._buffer, BATCH_SIZE, episode)
                        self.global_agent.update_parameters_by_worker_gradient(self.agent1, qf_loss, policy_loss, alpha_loss)
                    print(f'[Worker-{self._worker_id}] Episode #{episode}: {np.sum(mem1[:, 2])} {np.sum(mem2[:, 2])}')
                    if self._writer is not None:
                        ratings = reevaluate_ratings(ratings[0], ratings[1], info['win'] == 0)
                        returns = np.sum(mem1[:, 2])
                        if returns > best_score:
                            self.agent1.save(os.path.join(os.path.dirname(__file__), 'checkpoints', f'{ENVIRONMENT}-sac-ep-{episode}-score-{int(returns)}.ckpt'))
                            best_score = returns
                        self._writer.add_scalar('Reward', returns, episode)
                        self._writer.add_scalar('Rating', ratings[0], episode)
                        self.agent1.set_state_dict(self.global_agent.get_state_dict())
                    else:
                        self.load_learner_parameters()
                    break

        self._env.close()

    def load_learner_parameters(self):
        self.agent1.set_state_dict(self.global_agent.get_state_dict())
        self.agent2.set_state_dict(self.global_agent.get_state_dict())


def main():
    agent = Learner()
    agent.run()


if __name__ == "__main__":
    main()
