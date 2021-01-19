#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import abc
import random

import numpy as np


class ReplayMemory(abc.ABC):
    pass


class ReplayBuffer:

    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    pass