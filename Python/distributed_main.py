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
from utils.buffer import ReplayBuffer

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


SAMPLE_SIZE = 64
BATCH_SIZE = 32
CURRENT_EPISODE = 1
RESULTS = deque([0] * 10000, maxlen=10000)
RESULTS100 = deque([0] * 100, maxlen=100)
RESULTS1000 = deque([0] * 1000, maxlen=1000)
GLOBAL_WEIGHT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'a3c.h5')
# JOB_QUEUE = Queue()
REPLAY_MEMORY = ReplayBuffer()


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

        # _ = self.global_agent.get_action(np.random.uniform(-1, 1, (1, 64, 16)))

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
        global RESULTS

        epsilon = 1
        epsilon_discount = 0.001
        epsilon_minimum = 0.005

        while True:
            observations = []
            next_observations = []
            actions = []
            rewards = []
            dones = []

            observation = self.env.reset()
            # observation = np.stack([observation[0]] * SAMPLE_SIZE, axis=1)

            while True:
                observations.append(observation)

                if np.random.uniform(0, 1) > epsilon:
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

                # observation = np.append([next_observation[0, 0]], observation[0, :-1]).reshape((1, SAMPLE_SIZE, -1))
                next_observations.append(next_observation)
                rewards.append(reward)
                actions.append(action[0])
                dones.append(not done)

                if done:
                    # observations = np.squeeze(np.array(observations), axis=1)
                    # next_observations = np.squeeze(np.array(next_observations), axis=1)
                    # actions = np.array(actions)
                    # rewards = np.array(rewards)
                    # dones = np.array(dones)

                    with self.lock:
                        for obs, action, reward, obs_, done in zip(observations, actions, rewards, next_observations, dones):
                            REPLAY_MEMORY.append((obs, action, reward, obs_, done))

                    RESULTS.append(info['win'])
                    RESULTS100.append(info['win'])
                    RESULTS1000.append(info['win'])
                    returns = discount_rewards(rewards, dones)

                    # job = (observations, next_observations, actions, rewards, dones)
                    # JOB_QUEUE.put(job)

                    with self.lock:
                        episode = CURRENT_EPISODE
                        CURRENT_EPISODE += 1

                        # loss = self.global_agent.update(observations, actions, next_observations, rewards, dones)
                        if not args.torch:
                            loss = 0
                            # flag = False

                            batch = REPLAY_MEMORY.sample(BATCH_SIZE)
                            s, a, r, s_, d = [], [], [], [], []
                            for bs, ba, br, bs_, bd in batch:
                                s.append(bs)
                                a.append(ba)
                                r.append(br)
                                s_.append(bs_)
                                d.append(bd)

                            s = np.squeeze(np.array(s), axis=1)
                            a = np.array(a)
                            r = np.array(r)
                            s_ = np.squeeze(np.array(s_), axis=1)
                            d = np.array(d)
                            """
                            s = np.concatenate([b for b in batch[:, 0].squeeze()]).squeeze()
                            a = np.concatenate([b for b in batch[:, 1].squeeze()]).squeeze()
                            r = np.concatenate([b for b in batch[:, 2].squeeze()]).squeeze()
                            s_ = np.concatenate([b for b in batch[:, 3].squeeze()]).squeeze()
                            d = np.concatenate([b for b in batch[:, 4].squeeze()]).squeeze()
                            """
                            print('s.shape:', s.shape, a.shape, r.shape, s_.shape, d.shape)
                            print('---squeeze')

                            try:
                                with tf.device('/GPU:0'):
                                    loss = self.global_agent.update(s, a, s_, r, d)
                            except Exception as e:
                                sys.stderr.write('Exception: %s\n' % e)
                                break

                            """
                            for b in range(np.ceil(len(observations) / BATCH_SIZE).astype(np.uint8)):
                                batch_observations = observations[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
                                batch_actions = actions[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
                                batch_next_observations = next_observations[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
                                batch_rewards = rewards[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
                                batch_dones = dones[b*BATCH_SIZE:(b+1)*BATCH_SIZE]

                                try:
                                    with tf.device('/GPU:0'):
                                        loss = self.global_agent.update(batch_observations, batch_actions, batch_next_observations, batch_rewards, batch_dones)
                                    #with tf.device('/GPU:0'):
                                    #    loss = self.global_agent.update(observations, actions, next_observations, rewards, dones)
                                    #    del observations, actions, next_observations, rewards, dones
                                except Exception as e:
                                    sys.stderr.write('Exception: %s\n' % e)
                                    flag = True
                                    break

                            if flag:
                                break
                            """

                            del observations, actions, next_observations, rewards, dones

                        else:
                            loss = self.global_agent.update(observations, actions, next_observations, rewards, dones)
                        print('[%s] Episode %d: Loss: %f' % (datetime.now().isoformat(), episode, loss))

                        if episode % 100 == 0:
                            self.global_agent.save(GLOBAL_WEIGHT_PATH)

                        if not args.torch:
                            with TENSORBOARD_WRITER.as_default():
                                tf.summary.scalar('Reward', np.sum(returns), episode)
                                tf.summary.scalar('Loss', loss, episode)
                                tf.summary.scalar('Rate', np.mean(RESULTS), episode)
                                tf.summary.scalar('Rate/100', np.mean(RESULTS100), episode)
                                tf.summary.scalar('Rate/1000', np.mean(RESULTS1000), episode)
                        else:
                            TENSORBOARD_WRITER.add_scalar('Reward', np.sum(returns), episode)
                            TENSORBOARD_WRITER.add_scalar('Loss', loss, episode)
                            if len(RESULTS) >= 100:
                                TENSORBOARD_WRITER.add_scalar('Rate', np.mean(RESULTS), episode)

                        try:
                            self.agent.set_weights(self.global_agent.get_weights())
                        except Exception as e:
                            self.agent.load(GLOBAL_WEIGHT_PATH)

                    epsilon = max(epsilon - epsilon_discount, epsilon_minimum)

                    self.agent.reset()

                    break

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
