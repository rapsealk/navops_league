#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
from threading import Thread, Lock
from multiprocessing import cpu_count

import gym
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')  # softmax sum 1
cpu_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=cpu_devices, device_type='CPU')
# gpu_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=[], device_type='GPU')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=5e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
args = parser.parse_args()

CURRENT_EPISODE = 0


class Model(tf.keras.Model):

    def __init__(self, input_shape, n, learning_rate=5e-4):
        super(Model, self).__init__()

        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.pi = tf.keras.layers.Dense(n, activation='softmax')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.value = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.entropy_beta = 1e-2

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        policy = self.pi(x + 1e-8)
        x = self.dense3(x)
        value = self.value(x)
        return policy, value

    def compute_loss(self, actions, logits, advantages, values, targets):
        crossentropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, dtype=tf.uint8)
        policy_loss = crossentropy_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        mse = tf.keras.losses.Huber()
        value_loss = mse(targets, values)
        return policy_loss - self.entropy_beta * entropy + value_loss / 2

    def train(self, states, actions, advantages, targets):
        with tf.GradientTape() as tape:
            logits, values = self(states, training=True)
            loss = self.compute_loss(actions, logits, advantages, values, tf.stop_gradient(targets))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


class Agent:

    def __init__(self, env_name):
        env = gym.make(env_name)
        self.env_name = env_name
        input_shape = env.observation_space.shape
        n = env.action_space.n

        self.global_model = Model(input_shape, n)
        self.num_workers = cpu_count()

    def train(self, max_episodes=1_000_000):
        workers = []
        for i in range(self.num_workers):
            env = gym.make(self.env_name)
            workers.append(
                Worker(env, self.global_model, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class Worker(Thread):

    def __init__(self, env, global_model, max_episodes, gamma=0.99):
        Thread.__init__(self)
        # super(Worker, self).__init__()

        self.lock = Lock()
        self.env = env
        self.input_shape = env.observation_space.shape
        self.n = env.action_space.n

        self.max_episodes = max_episodes
        self.global_model = global_model
        self.model = Model(self.input_shape, self.n)
        self.gamma = gamma

        self.model.set_weights(self.global_model.get_weights())

    def n_step_td_target(self, rewards, next_value, dones):
        td_targets = np.zeros_like(rewards)

        for t in reversed(range(td_targets.shape[0]-1)):
            td_targets[t] = rewards[t] + (1 - dones[t]) * self.gamma * td_targets[t+1]

        return td_targets

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def run(self):
        global CURRENT_EPISODE

        while self.max_episodes >= CURRENT_EPISODE:
            states = []
            actions = []
            rewards = []
            dones = []
            total_reward = 0

            state = self.env.reset()

            while True:
                policy, value = self.model(state[np.newaxis, :])
                action = np.random.choice(self.n, p=policy[0])

                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                if done or len(states) >= args.update_interval:
                    states = np.array(states)[:, np.newaxis]
                    actions = np.array(actions)[:, np.newaxis]
                    rewards = np.array(rewards)[:, np.newaxis]
                    dones = np.asarray(dones)

                    _, next_value = self.model(next_state[np.newaxis, :])
                    td_targets = self.n_step_td_target(rewards, next_value, dones)
                    advantages = td_targets - rewards   # self.model(states)

                    with self.lock:
                        loss = self.global_model.train(states, actions, advantages, td_targets)
                        # print('Loss: %f' % loss)
                        self.model.set_weights(self.global_model.get_weights())

                    states = []
                    actions = []
                    rewards = []
                    dones = []

                total_reward += np.sum(rewards)
                state = next_state

                if done:
                    break

            print('Episode %d: %f' % (CURRENT_EPISODE, total_reward))
            with TENSORBOARD_WRITER.as_default():
                tf.summary.scalar('Reward', total_reward, CURRENT_EPISODE)
            CURRENT_EPISODE += 1


if __name__ == "__main__":
    global TENSORBOARD_WRITER

    TENSORBOARD_WRITER = tf.summary.create_file_writer(
        os.path.join(os.path.dirname(__file__), 'summary'))
    env_name = 'CartPole-v1'
    agent = Agent(env_name)
    agent.train()
