#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse

import gym
import gym_rimpac   # noqa: F401
import numpy as np

from models.pytorch_impl import PPOAgent
from utils import discount_rewards
from demos import DemoLoader

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4096)
parser.add_argument('--seq-len', type=int, default=64)
args = parser.parse_args()


def generate_observation(observations):
    observation_shape = observations[0].shape[0]
    new_observations = []
    next_obs = np.zeros((args.seq_len * observation_shape,))
    for obs in observations:
        next_obs = np.concatenate([next_obs[observation_shape:], obs])
        new_observations.append(next_obs)
    new_obs = np.array(new_observations)
    return new_obs


def main():
    env = gym.make('RimpacDiscrete-v0', mock=True)
    print(f'env(obs={env.observation_space.shape}, act={env.action_space.n})')
    agent = PPOAgent(
        env.observation_space.shape[0] * args.seq_len,
        env.action_space.n,
        learning_rate=1e-3,
        cuda=False
    )
    s, a, r, d = DemoLoader().load()
    s = generate_observation(s)
    dd = np.where(d == 1.0)[0]
    endpoint = dd[-1] + 1
    s, a, r, d = s[:endpoint], a[:endpoint], r[:endpoint], d[:endpoint]
    start = 0
    for i in range(len(dd)):
        r[start:dd[i]] = discount_rewards(r[start:dd[i]])
        start = dd[i] + 1

    for _ in range(3):
        loss = 0
        for b in range(len(s) // args.batch_size + 1):
            x, y = (b * args.batch_size, (b + 1) * args.batch_size)
            loss += agent.supervised_learning(s[x:y], a[x:y], r[x:y])
        print(f'[{_}] Loss: {loss}')

    agent.save(os.path.join(os.path.dirname(__file__), 'checkpoints', 'pretrained', f'supervised-seq_{args.seq_len}.ckpt'))


if __name__ == "__main__":
    main()
