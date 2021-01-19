#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime
from itertools import count

import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import SoftActorCriticAgent, ReplayBuffer


def epsilon():
    eps = 1.0
    for i in count(1):
        yield eps
    # eps = epsilon()
    # next(eps)


def main():
    env = gym.make('MountainCarContinuous-v0')
    writer = SummaryWriter('runs/{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    agent1 = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
    memory = ReplayBuffer(1000000)

    for episode in count(0):
        observation = env.reset()
        total_rewards = 0
        # rewards = []

        while True:
            """epsilon
            action = np.random.uniform(0.0, 1.0, size=(2, 4))
            action[:, :2] = (action[:, :2] - 0.5) * 2
            """
            action = agent1.select_action(observation)
            next_observation, reward, done, info = env.step(action)

            env.render()

            # rewards.append(reward)
            memory.push(observation, action, reward, next_observation, done)

            observation = next_observation
            total_rewards += reward

            if len(memory) % 128 == 0:
                n = len(memory) // 128
                # qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
                q1_1, q2_1, pi_1, alpha_1, alpha_tlog_1 = agent1.update_parameters(memory, 128, n)

                # SummaryWriter
                writer.add_scalar('loss/b/critic_1', q1_1, n)
                writer.add_scalar('loss/b/critic_2', q2_1, n)
                writer.add_scalar('loss/b/policy', pi_1, n)
                writer.add_scalar('loss/b/entropy_loss', alpha_1, n)
                writer.add_scalar('entropy_temprature/b/alpha', alpha_tlog_1, n)

            if done:
                writer.add_scalar('reward', total_rewards, episode)
                print('[{}] Episode {}: {}'.format(datetime.now().isoformat(), episode, total_rewards))
                total_rewards = 0
                break

    env.close()


if __name__ == "__main__":
    main()
