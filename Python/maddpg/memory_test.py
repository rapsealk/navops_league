#!/usr/bin/python3
# -*- coding: utf-8 -*-
import unittest
import time
import random

import numpy as np
import torch

from memory import _MongoReplayBuffer as MongoReplayBuffer


class MongoReplayBufferTestCase(unittest.TestCase):

    def setUp(self):
        self.memory = MongoReplayBuffer()
        print('Add data..', end=' ')
        time_ = time.time()
        for _ in range(2048):
            self.memory.push(*self._generate_data())
        print(f'Done! - {time.time() - time_}s')

    def test_sample(self):
        time_ = time.time()
        sample = self.memory.sample(batch_size=1024)
        print(f'Test sample: {time.time() - time_}s')   # 50.707s
        print('sample:', sample[0])

    def tearDown(self):
        self.memory.clear()

    def _generate_data(self, n=3):
        observations = np.random.uniform(-1.0, 1.0, (64, 3, 118))
        actions = np.random.randint(0, 7)
        next_observations = np.concatenate([observations[1:], np.random.uniform(-1.0, 1.0, (1, 3, 118))])
        rewards = np.random.normal(0.0, 1.0, (3,))
        h_ins = [torch.randn(4, 1, 8) for _ in range(n)]
        # h_outs = [torch.randn(4, 1, 8) for _ in range(n)]
        done = random.choice([False, True])

        return (observations, actions, next_observations, rewards, h_ins, done)


if __name__ == "__main__":
    unittest.main()
