#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import time
from datetime import datetime
from collections import namedtuple

import numpy as np

from environment import UnityEnvironmentImpl


def main():
    env = UnityEnvironmentImpl()

    observation = env.reset()

    blue = 0
    red = 0

    while True:
        action = np.zeros(shape=(2, 6))
        for i in range(action.shape[0]):
            action[i, np.random.randint(0, 6)] = 1

        observation, reward, done = env.step(action)
        # print('observation:', observation)

        blue += reward[0]
        red += reward[1]

        if done:
            print('reward:', reward, reward.shape, blue, red)
            print('done:', done)
            break

    env.close()


if __name__ == "__main__":
    main()
