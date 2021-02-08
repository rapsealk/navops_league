#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime

import gym
import gym_rimpac   # noqa: F401
import numpy as np

from models.pytorch_impl.model import SoftActorCriticAgent
from memory import MongoReplayBuffer as ReplayBuffer


def process_raw_observation(next_obs, *obs):
    next_obs = np.expand_dims(next_obs, axis=1)
    return (
        np.concatenate((
            obs[i][1:],
            np.expand_dims(np.concatenate((obs[i][-1, 1:], next_obs[i])), axis=0)
        ))
        for i in range(len(obs))
    )


def main():
    env = gym.make('Rimpac-v0', mock=True)
    print(env.observation_space.shape)
    print(env.action_space.shape)

    input_size = env.observation_space.shape[0]
    batch_size = 16
    seq_len = 4
    agent = SoftActorCriticAgent(
        input_size=input_size,
        output_size=env.action_space.shape[0],
        hidden_size=128,
        action_space=env.action_space,
        num_layers=64,
        batch_size=batch_size,
        force_cpu=True
    )
    buffer = ReplayBuffer(1000)

    obs_batch1 = np.zeros((batch_size, seq_len, *env.observation_space.shape))
    obs_batch2 = np.zeros((batch_size, seq_len, *env.observation_space.shape))
    obs = env.reset()
    obs_batch1, obs_batch2 = process_raw_observation(obs, obs_batch1, obs_batch2)

    memory1, memory2 = [], []

    for _ in range(batch_size*2):
        action1 = agent.get_action(obs_batch1)
        action2 = agent.get_action(obs_batch2)
        action = np.concatenate((action1[-1, -1:], action2[-1, -1:]))

        next_obs, reward, done, info = env.step(action)
        print(f'[{datetime.now().isoformat()}] obs: {obs.shape} ({_})')
        next_obs_batch1, next_obs_batch2 = process_raw_observation(next_obs, obs_batch1, obs_batch2)

        memory1.append((obs_batch1[-1], action1[-1], reward[0], next_obs_batch1[-1], done))
        memory2.append((obs_batch2[-1], action2[-1], reward[1], next_obs_batch2[-1], done))

        obs_batch1, obs_batch2 = next_obs_batch1, next_obs_batch2

    memory1 = np.array(memory1)
    memory1[:, 2] = 1
    memory2 = np.array(memory2)
    memory2[:, 2] = 0

    for s, a, r, s_, d in np.concatenate((memory1, memory2)):
        buffer.push(s, a, r, s_, d)
    if len(buffer) >= batch_size:
        qf1, qf2, pi = agent.update_parameters(buffer, batch_size)
        print(f'qf1: {qf1}, qf2: {qf2}, pi: {pi}')


if __name__ == "__main__":
    main()
