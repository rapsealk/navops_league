#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import socket
import argparse
import time
import queue
from collections import deque
from datetime import datetime
from threading import Thread, Lock
from multiprocessing import cpu_count, Queue

import numpy as np
import tensorflow as tf

from environment import UnityEnvironmentImpl

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default=None)
# parser.add_argument('--tensorflow', action='store_true', default=False)
parser.add_argument('--torch', action='store_true', default=False)
parser.add_argument('--core', type=int, default=cpu_count())
parser.add_argument('--cpu', action='store_true', default=False)
args = parser.parse_args()

if not args.torch:
    import tensorflow as tf
    from models.tensorflow_impl.ppo_lstm import Agent

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    else:
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
else:
    from torch.utils.tensorboard import SummaryWriter
    from models.pytorch_impl.ppo_lstm import ProximalPolicyOptimizationAgent as Agent


BATCH_SIZE = 128
SAMPLE_SIZE = 64
CURRENT_EPISODE = 0
EPSILON = 1
EPSILON_DISCOUNT = 0.01
EPSILON_MINIMUM = 0.005
RESULTS = deque([], maxlen=100)
# JOB_QUEUE = Queue()


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
        self.global_agent = Agent(n=6)
        self.global_agent.load()
        self.num_workers = args.core    # cpu_count()

        self.queue = Queue()

    def train(self, max_episodes=1000000):
        workers = []
        for i in range(self.num_workers):
            workers.append(Worker(i, get_available_port(), self.global_agent, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

    def run(self):
        while True:
            try:
                job = JOB_QUEUE.get_nowait()    # get()
            except queue.Empty:
                time.sleep(1)
                continue


class Worker(Thread):

    def __init__(self, worker_id, base_port, global_agent, max_episodes, gamma=0.99):
        Thread.__init__(self, daemon=True)

        self.lock = Lock()
        self.env = UnityEnvironmentImpl(worker_id=worker_id, base_port=base_port)

        self.global_agent = global_agent
        self.max_episodes = max_episodes
        self.agent = Agent(n=self.env.action_space)
        self.gamma = gamma

        self.agent.set_weights(global_agent.get_weights())

    def run(self):
        global CURRENT_EPISODE
        global EPSILON
        global RESULTS

        while True:
            observations = []
            next_observations = []
            actions = []
            rewards = []
            dones = []

            observation = self.env.reset()
            observation = np.stack([observation[0]] * SAMPLE_SIZE, axis=1)

            total_loss = 0
            total_rewards = 0

            while True:
                observations.append(observation)

                if np.random.uniform(0, 1) > EPSILON:
                    try:
                        # policy = agent.get_action(observation)
                        if not args.torch:
                            with tf.device('/CPU:0'):
                                policy = agent.get_action(observation)
                        # print('Policy:', np.squeeze(policy[0]), np.squeeze(policy[-1]))
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

                observation = np.append([next_observation[0, 0]], observation[0, :-1]).reshape((1, SAMPLE_SIZE, -1))
                next_observations.append(observation)
                rewards.append(reward)
                actions.append(action[0])
                dones.append(not done)

                if done or len(observation) == BATCH_SIZE:
                    observations = np.squeeze(np.array(observations), axis=1)
                    next_observations = np.squeeze(np.array(next_observations), axis=1)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    dones = np.array(dones)

                    RESULTS.append(info['win'])
                    returns = discount_rewards(rewards, dones)

                    # job = (observations, next_observations, actions, rewards, dones)
                    # JOB_QUEUE.put(job)

                    with self.lock:
                        # loss = self.global_agent.update(observations, actions, next_observations, rewards, dones)
                        if not args.torch:
                            loss = 0
                            try:
                                with tf.device('/GPU:0'):
                                    loss = self.global_agent.update(observations, actions, next_observations, rewards, dones)
                                del observations, actions, next_observations, rewards, dones
                            except:
                                break
                        else:
                            try:
                                loss = self.global_agent.update(observations, actions, next_observations, rewards, dones)
                            except RuntimeError as e:
                                sys.stderr.write('RuntimeError: %s\n' % e)
                                break
                        print('[%s] Episode %d: Loss: %f' % (datetime.now().isoformat(), CURRENT_EPISODE, loss))

                        if CURRENT_EPISODE % 100 == 0:
                            self.global_agent.save()

                        total_loss += loss
                        total_rewards += np.sum(returns)

                    if done:
                        if not args.torch:
                            with TENSORBOARD_WRITER.as_default():
                                tf.summary.scalar('Reward', total_rewards, CURRENT_EPISODE)
                                tf.summary.scalar('Loss', total_loss, CURRENT_EPISODE)
                                if len(RESULTS) >= 100:
                                    tf.summary.scalar('Rate', np.mean(RESULTS), CURRENT_EPISODE)
                        else:
                            TENSORBOARD_WRITER.add_scalar('Reward', total_rewards, CURRENT_EPISODE)
                            TENSORBOARD_WRITER.add_scalar('Loss', total_loss, CURRENT_EPISODE)
                            if len(RESULTS) >= 100:
                                TENSORBOARD_WRITER.add_scalar('Rate', np.mean(RESULTS), CURRENT_EPISODE)
                        CURRENT_EPISODE += 1

                        EPSILON = max(EPSILON - EPSILON_DISCOUNT, EPSILON_MINIMUM)

                        observation = self.env.reset()
                        observation = np.stack([observation[0]] * SAMPLE_SIZE, axis=1)

                        break

                    # break

        self.env.close()


if __name__ == "__main__":
    global TENSORBOARD_WRITER

    if not args.torch:
        logdir = args.tag or datetime.now().isoformat().replace(':', '').split('.')[0]
        TENSORBOARD_WRITER = tf.summary.create_file_writer(os.path.join(os.path.dirname(__file__), 'summary', 'distributed', logdir))
    else:
        TENSORBOARD_WRITER = SummaryWriter()

    agent = Learner()
    agent.train()
