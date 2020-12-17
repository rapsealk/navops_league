#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import random


class ReplayBuffer:

    def __init__(self, capacity=1000000):
        self.memory = []
        self.capacity = capacity

    def append(self, data):
        self.memory.append(data)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, n):
        return random.sample(self.memory, n)


if __name__ == "__main__":
    pass
