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


def epsilon():
    eps = 1.0
    eps_discount = 0.001
    eps_discount_step = 1000
    eps_minimum = 0.005
    for i in count(1):
        if i % eps_discount_step == 0:
            eps = max(eps - eps_discount, eps_minimum)
        yield eps


def main():
    env = gym.make('Rimpac-v0', no_graphics=args.no_graphics)
    writer = SummaryWriter('runs/{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    agent1 = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
    agent2 = SoftActorCriticAgent(env.observation_space.shape[0], env.action_space)
    memory = ReplayBuffer(1000000)

    eps = epsilon()
    step = 0
    batch_size = 64

    for episode in count(1):
        observation = env.reset()
        total_rewards = [0, 0]

        while True:
            if np.random.random() > next(eps):
                actions = (agent1.select_action(observation[0])[0],
                           agent2.select_action(observation[1])[0])
            else:
                actions = (np.random.uniform(0.0, 1.0, size=(2,)+env.action_space.shape))
                actions[:, :2] = (actions[:, :2] - 0.5) * 2
                actions[:, 8:] = (actions[:, 8:] - 0.5) * 2
            action = np.stack(actions, axis=0)  # .squeeze(0)
            next_observation, reward, done, info = env.step(action)

            # rewards.append(reward)
            # print('observation:', observation.shape, next_observation.shape, done)
            memory.push(observation[0, 0], actions[0], reward[0], next_observation[0, 0], done)
            memory.push(observation[1, 0], actions[1], reward[1], next_observation[1, 0], done)

            #print('reward:', reward.shape, done)
            # if 0 in reward.shape:
            #     print('- reward:', reward, done)
            total_rewards[0] += reward[0].item()
            total_rewards[1] += reward[1].item()

            observation = next_observation

            if len(memory) % batch_size == 0:
                step += 1
                # qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
                q1_1, q2_1, pi_1, alpha_1, alpha_tlog_1 = agent1.update_parameters(memory, batch_size, step)
                q1_2, q2_2, pi_2, alpha_2, alpha_tlog_2 = agent2.update_parameters(memory, batch_size, step)

                # SummaryWriter
                writer.add_scalar('loss/b/critic_1', q1_1, step)
                writer.add_scalar('loss/b/critic_2', q2_1, step)
                writer.add_scalar('loss/b/policy', pi_1, step)
                writer.add_scalar('loss/b/entropy_loss', alpha_1, step)
                writer.add_scalar('entropy_temprature/b/alpha', alpha_tlog_1, step)
                writer.add_scalar('loss/r/critic_1', q1_2, step)
                writer.add_scalar('loss/r/critic_2', q2_2, step)
                writer.add_scalar('loss/r/policy', pi_1, step)
                writer.add_scalar('loss/r/entropy_loss', alpha_2, step)
                writer.add_scalar('entropy_temprature/r/alpha', alpha_tlog_2, step)

                print('[{}] ({}) {}, {}, {}, {}, {}'.format(datetime.now().isoformat(), step, q1_1, q2_1, pi_1, alpha_1, alpha_tlog_1))

            if done:
                writer.add_scalar('reward/b', total_rewards[0], episode)
                writer.add_scalar('reward/r', total_rewards[1], episode)
                break

    env.close()


if __name__ == "__main__":
    main()
