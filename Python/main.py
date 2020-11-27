#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
from itertools import count

import tensorflow as tf
import numpy as np

from environment import UnityEnvironmentImpl
from models.tensorflow_impl import ActorCriticLSTM

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', type=int, default=20800)
args = parser.parse_args()


def discount_rewards(rewards, dones, gamma=0.99):
    rewards = np.array(rewards, dtype=np.float32)
    returns = np.append(np.zeros_like(rewards), np.zeros((1, 1))).reshape((-1, 1))
    for t in reversed(range(rewards.shape[0])):
        returns[t] = rewards[t] + gamma * returns[t+1] * (1 - dones[t])
    # print('__discount_rewards: returns.shape:', returns.shape, ' rewards.shape:', rewards.shape)
    return returns[:-1]


def main():
    writer = tf.summary.create_file_writer(
        os.path.join(os.path.dirname(__file__), 'summary')
    )

    n = 1
    env = UnityEnvironmentImpl(n, worker_id=args.port)

    epsilon = 1
    epsilon_discount = 0.01

    blue_model = ActorCriticLSTM('actorcritic_lstm_blue')
    # red_model = ActorCriticLSTM('actorcritic_lstm_red')

    for episode in count(0):
        blue_observations = []
        blue_next_observations = []
        blue_actions = []
        blue_rewards = []
        """
        red_observations = []
        red_next_observations = []
        red_actions = []
        red_rewards = []
        """
        dones = []

        observation = env.reset()
        blue_observation = np.stack([observation[0], observation[0], observation[0], observation[0]],
                                    axis=0)
        """
        red_observation = np.stack([observation[1], observation[1], observation[1], observation[1]],
                                   axis=0)
        """

        while True:
            blue_observations.append(blue_observation)
            # red_observations.append(red_observation)

            if np.random.uniform(0, 1) > epsilon:
                try:
                    blue_pi, _ = blue_model(blue_observation)
                    blue_pi = np.squeeze(blue_pi[0])
                    blue_pi = np.random.choice(blue_pi.shape[-1], 1, p=blue_pi)[0]
                    blue_action = np.zeros(shape=(n, 6))
                    for i in range(n):
                        blue_action[i, blue_pi[i]] = 1
                    """
                    red_pi, _ = red_model(red_observation)
                    red_pi = np.squeeze(red_pi[0])
                    red_pi = np.random.choice(red_pi.shape[-1], 1, p=red_pi)[0]
                    red_action = np.zeros(shape=(1, 6))
                    red_action[0, red_pi] = 1
                    """
                except:
                    blue_action = np.zeros(shape=(n, 6))
                    for _ in range(n):
                        blue_action[_, np.random.randint(0, 6)] = 1
                    """
                    red_action = np.zeros(shape=(1, 6))
                    red_action[0, np.random.randint(0, 6)] = 1
                    """
            else:
                blue_action = np.zeros(shape=(n, 6))
                for _ in range(n):
                    blue_action[_, np.random.randint(0, 6)] = 1
                """
                red_action = np.zeros(shape=(1, 6))
                red_action[0, np.random.randint(0, 6)] = 1
                """

            # observation, reward, done = env.step([blue_action, red_action])
            print('action.shape:', blue_action.shape)
            observation, reward, done = env.step(blue_action)
            print('observation.shape:', observation.shape, '/ reward.shape:', reward.shape)

            blue_observation = np.append([observation[0]], blue_observation[:3]).reshape((4, 1, -1))
            # red_observation = np.append([observation[1]], red_observation[:3]).reshape((4, 1, -1))
            blue_next_observations.append(blue_observation)
            # red_next_observations.append(red_observation)
            blue_rewards.append(reward)
            # red_rewards.append(reward[1])
            blue_actions.append(blue_action)
            # red_actions.append(red_action)
            dones.append(not done)

            # print(datetime.now().isoformat())

            if done:
                blue_returns = discount_rewards(blue_rewards, dones)
                # red_returns = discount_rewards(red_rewards, dones)
                with writer.as_default():
                    tf.summary.scalar('Blue Reward', np.sum(blue_returns), episode)
                    # tf.summary.scalar('Red Reward', np.sum(red_returns), episode)

                # blue_model.reset()
                # red_model.reset()

                blue_loss = blue_model.train(np.array(blue_observations),
                                             np.array(blue_actions),
                                             np.array(blue_returns),
                                             np.array(blue_next_observations),
                                             np.array(dones))
                """
                red_loss = red_model.train(np.array(red_observations),
                                           np.array(red_actions),
                                           np.array(red_returns),
                                           np.array(red_next_observations),
                                           np.array(dones))
                """

                with writer.as_default():
                    tf.summary.scalar('Blue Loss', np.sum(blue_loss), episode)
                    # tf.summary.scalar('Red Loss', np.sum(red_loss), episode)

                epsilon = max(epsilon - epsilon_discount, 0.01)

                break

                # Reset
                blue_observations = []
                blue_next_observations = []
                blue_actions = []
                blue_rewards = []
                """
                red_observations = []
                red_next_observations = []
                red_actions = []
                red_rewards = []
                """
                dones = []

                observation = env.reset()
                blue_observation = np.stack([observation[0], observation[0], observation[0], observation[0]],
                                            axis=0)
                """
                red_observation = np.stack([observation[1], observation[1], observation[1], observation[1]],
                                           axis=0)
                """

    env.close()


if __name__ == "__main__":
    main()
