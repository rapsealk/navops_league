#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
# from datetime import datetime
from itertools import count

import gym
import gym_rimpac   # noqa: F401
import numpy as np

from models.pytorch_impl import SoftActorCriticAgent, ReplayBuffer

ENVIRONMENT = 'Mock-Rimpac-v0'
BATCH_SIZE = 16
TIME_SEQUENCE = 4


def process_raw_observation(obs1, obs2, next_obs):
    next_obs = np.expand_dims(next_obs, axis=1)
    next_obs1 = np.concatenate((
        obs1[1:],
        np.expand_dims(np.concatenate((obs1[-1, 1:], next_obs[0])), axis=0)
    ))
    next_obs2 = np.concatenate((
        obs2[1:],
        np.expand_dims(np.concatenate((obs2[-1, 1:], next_obs[1])), axis=0)
    ))
    return next_obs1, next_obs2


if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    agent = SoftActorCriticAgent(
        env.observation_space.shape[0],
        env.action_space,
        hidden_dim=64,
        num_layers=32,
        batch_size=BATCH_SIZE
    )
    buffer = ReplayBuffer(100000)

    for episode in count(1):
        obs_batch1 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *env.observation_space.shape))  # 0.2 MB
        obs_batch2 = np.zeros((BATCH_SIZE, TIME_SEQUENCE, *env.observation_space.shape))

        obs = env.reset()
        # print('obs.shape:', obs.shape)
        obs_batch1, obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, obs)
        # print('obs1.shape:', obs_batch1.shape)

        for _ in range(BATCH_SIZE):
            print('obs_batch:', obs_batch1.shape)
            action1 = agent.select_action(obs_batch1)
            # print('action1.shape:', action1.shape)
            action2 = agent.select_action(obs_batch2)
            # print('action2.shape:', action2.shape)

            # print('action1.shape:', action1.shape)
            # print('action1[-1, -1].shape:', action1[-1, -1:].shape)
            actual_action = np.concatenate((action1[-1, -1:], action2[-1, -1:]))
            print('action:', actual_action.shape)

            # print('actual_action:', actual_action.shape)
            next_obs, reward, done, _ = env.step(actual_action)
            # print('reward:', reward)
            # print('done:', done)

            next_obs_batch1, next_obs_batch2 = process_raw_observation(obs_batch1, obs_batch2, next_obs)
            # print('next_obs1.shape:', obs_batch1.shape)

            buffer.push(obs_batch1[-1], action1[-1], reward[0], next_obs_batch1[-1], done)
            buffer.push(obs_batch2[-1], action2[-1], reward[1], next_obs_batch2[-1], done)


        # Train
        """
        q_loss, pi_loss, alpha_loss = agent.compute_gradient(buffer, BATCH_SIZE, episode)
        print(f'q_loss: {q_loss}, pi_loss: {pi_loss}, alpha_loss: {alpha_loss}')
        agent.descent_gradient(agent, q_loss, pi_loss, episode)
        """
        s, a, r, s_, dones = agent.get_trajectory_batch(buffer, BATCH_SIZE)
        q_loss = agent.get_value_gradient(s, a, r, s_, dones)
        agent.descent_gradient(local=agent, q_loss=q_loss)
        policy_loss = agent.get_policy_gradient(s, a, r, s_, dones)
        agent.descent_gradient(local=agent, pi_loss=policy_loss)

        print(f'memory:', sys.getsizeof(buffer))

        env.close()
        break
