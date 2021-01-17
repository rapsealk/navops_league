#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from itertools import count

import numpy as np
import gym
import gym_rimpac   # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import SoftActorCriticAgent, ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--no-graphics', action='store_true', default=False)
args = parser.parse_args()


def main():
    env = gym.make('Rimpac-v0', no_graphics=args.no_graphics)
    writer = SummaryWriter('runs/{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    agent1 = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
    agent2 = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
    memory = ReplayBuffer(1000000)

    for episode in count(0):
        observation = env.reset()
        # rewards = []

        while True:
            """epsilon
            action = np.random.uniform(0.0, 1.0, size=(2, 4))
            action[:, :2] = (action[:, :2] - 0.5) * 2
            """
            actions = (
                agent1.select_action(observation[0])[0],
                agent2.select_action(observation[1])[0]
            )
            action = np.stack(actions, axis=0)  # .squeeze(0)
            next_observation, reward, done, info = env.step(action)

            # rewards.append(reward)
            memory.push(observation[0], actions[0], reward[0], next_observation[0], done)
            memory.push(observation[1], actions[1], reward[1], next_observation[1], done)

            if len(memory) % 128 == 0:
                n = len(memory) // 128
                # qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
                q1_1, q2_1, pi_1, alpha_1, alpha_tlog_1 = agent1.update_parameters(memory, 128, n)
                q1_2, q2_2, pi_2, alpha_2, alpha_tlog_2 = agent2.update_parameters(memory, 128, n)

                # SummaryWriter
                writer.add_scalar('loss/b/critic_1', q1_1, n)
                writer.add_scalar('loss/b/critic_2', q2_1, n)
                writer.add_scalar('loss/b/policy', pi_1, n)
                writer.add_scalar('loss/b/entropy_loss', alpha_1, n)
                writer.add_scalar('entropy_temprature/b/alpha', alpha_tlog_1, n)
                writer.add_scalar('loss/r/critic_1', q1_2, n)
                writer.add_scalar('loss/r/critic_2', q2_2, n)
                writer.add_scalar('loss/r/policy', pi_1, n)
                writer.add_scalar('loss/r/entropy_loss', alpha_2, n)
                writer.add_scalar('entropy_temprature/r/alpha', alpha_tlog_2, n)

            if done:
                # print('rewards:', np.sum(rewards))
                # agent1.update_parameters(memory, 128, episode)
                # agent2.update_parameters(memory, 128, episode)
                break

    env.close()


if __name__ == "__main__":
    main()
