#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from itertools import count

import numpy as np
import gym
import gym_rimpac   # noqa: F401


def main():
    env = gym.make('Rimpac-v0')

    for episode in count(0):
        observation = env.reset()
        rewards = []
        print('reset:', observation.shape)

        while True:
            action = np.random.uniform(0.0, 1.0, size=(2, 4))
            action[:, :2] = (action[:, :2] - 0.5) * 2
            next_observation, reward, done, info = env.step(action)

            rewards.append(reward)

            if done:
                print('rewards:', np.sum(rewards))
                break

        break

    env.close()


if __name__ == "__main__":
    main()
