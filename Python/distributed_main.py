#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import socket
from collections import deque
from threading import Thread, Lock
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from environment import UnityEnvironmentImpl
from models.tensorflow_impl.ppo_lstm import Agent

CURRENT_EPISODE = 0


def get_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def discount_rewards(rewards, dones, gamma=0.99):
    rewards = np.array(rewards, dtype=np.float32)
    returns = np.append(np.zeros_like(rewards), np.zeros((1, 1))).reshape((-1, 1))
    for t in reversed(range(rewards.shape[0])):
        returns[t] = rewards[t] + gamma * returns[t+1] * (1 - dones[t])
    return returns[:-1]


class Learner:

    def __init__(self):
        env = UnityEnvironmentImpl()
        self.global_agent = Agent(n=env.action_space)
        self.num_workers = cpu_count()

    def train(self, max_episodes=1000000):
        workers = []
        for i in range(self.num_workers):
            workers.append(Worker(get_available_port(), self.global_agent, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class Worker(Thread):

    def __init__(self, worker_id, global_agent, max_episodes, gamma=0.99):
        Thread.__init__(self)

        self.lock = Lock()
        self.env = UnityEnvironmentImpl(worker_id=worker_id)

        self.global_agent = global_agent
        self.max_episodes = max_episodes
        self.agent = Agent(n=self.env.action_space)
        self.gamma = gamma

        self.agent.set_weights(global_agent.get_weights())

    def run(self):
        global CURRENT_EPISODE

        epsilon = 1
        epsilon_discount = 0.01
        epsilon_minimum = 0.01

        results = deque([0] * 99, maxlen=100)

        while True:
            observations = []
            next_observations = []
            actions = []
            rewards = []
            dones = []

            observation = self.env.reset()
            observation = np.stack([observation[0], observation[0], observation[0], observation[0]],
                                   axis=1)

            while True:
                observations.append(observation)

                if np.random.uniform(0, 1) > epsilon:
                    try:
                        policy = agent.get_action(observation)
                        policy = np.squeeze(policy[0])
                        policy = np.random.choice(policy.shape[-1], 1, p=policy)[0]
                        action = np.zeros(shape=(1, self.env.action_space))
                        action[0, policy] = 1
                    except:
                        action = np.zeros(shape=(1, self.env.action_space))
                        action[0, np.random.randint(0, self.env.action_space)] = 1
                else:
                    action = np.zeros(shape=(1, self.env.action_space))
                    action[0, np.random.randint(0, self.env.action_space)] = 1

                next_observation, reward, done, info = self.env.step(action)

                observation = np.append([next_observation[0, 0]], observation[0, :-1]).reshape((1, 4, -1))
                next_observations.append(observation)
                rewards.append(reward)
                actions.append(action[0])
                dones.append(not done)

                if done:
                    observations = np.squeeze(np.array(observations), axis=1)
                    next_observations = np.squeeze(np.array(next_observations), axis=1)

                    results.append(info['win'])
                    returns = discount_rewards(rewards, dones)

                    with self.lock:
                        loss = self.global_agent.update(observations, actions, next_observations, rewards, dones)
                        print('Episode %d: Loss: %f' % (CURRENT_EPISODE, loss))
                        self.global_agent.save()

                    with TENSORBOARD_WRITER.as_default():
                        tf.summary.scalar('Reward', np.sum(returns), CURRENT_EPISODE)
                        tf.summary.scalar('Loss', loss, CURRENT_EPISODE)
                        tf.summary.scalar('Rate', np.mean(results), CURRENT_EPISODE)
                    CURRENT_EPISODE += 1

                    epsilon = max(epsilon - epsilon_discount, epsilon_minimum)

                    break

        self.env.close()


if __name__ == "__main__":
    global TENSORBOARD_WRITER

    TENSORBOARD_WRITER = tf.summary.create_file_writer(os.path.join(os.path.dirname(__file__), 'summary'))

    agent = Learner()
    agent.train()
