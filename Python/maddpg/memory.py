#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import random
from collections import deque

import numpy as np
import torch
import pymongo
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import generate_id

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)
    MONGO_USERNAME = config["mongo"]["username"]
    MONGO_PASSWORD = config["mongo"]["password"]


class ReplayBuffer:

    def __init__(self, capacity=1_000_000, seed=0):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, *args):
        self._buffer.append(args)

    def sample(self, batch_size, on_policy=False):
        batch_size = min(batch_size, len(self._buffer))
        if on_policy:
            return [self._buffer[-1]]
        return random.sample(self._buffer, batch_size)

    def extend(self, items):
        self._buffer.extend(items)

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    @property
    def items(self):
        return tuple(self._buffer)


class MongoReplayBuffer:

    def __init__(self, username=MONGO_USERNAME, password=MONGO_PASSWORD, port=27017):
        self._id = generate_id()
        # mongodb://root:1111@localhost:27017/
        client = pymongo.MongoClient(f'mongodb://{username}:{password}@localhost:{port}/')
        database = client["navops"]
        self._collection = database["buffer"]

    def push(self, *args):
        # episode_buffer.push(observations, actions, next_observationss, rewards, h_ins, h_outs, done)
        for arg in args:
            pass

        # _ = self._collection.insert_one()

    def sample(self, batch_size, on_policy=False):
        pipeline = [
            {"$sample": {"size": batch_size}}
        ]
        batch = map(self._decode, self._collection.aggregate(pipeline))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        del batch
        return state, action, reward, next_state, done
        """
        batch_size = min(batch_size, len(self._buffer))
        if on_policy:
            return [self._buffer[-1]]
        return random.sample(self._buffer, batch_size)
        """

    def extend(self, items):
        pass

    def clear(self):
        pass    # self._collection.drop()

    def __len__(self):
        return self._collection.estimated_document_count()

    """
    def _decode(self, row):
        return (
            np.array(row["state"]),
            np.array(row["action"]),
            row["reward"],
            np.array(row["next_state"]),
            row["done"]
        )
    """

    def _torch_to_list(self, torch_):
        item = torch_.numpy().tolist()
        return item

    def _list_to_numpy(self, list_):
        item = np.array(list_, dtype=np.float32)
        return item

    def _numpy_to_tensor(self, numpy_):
        item = torch.from_array(numpy_).float()
        return item

    def _list_to_tensor(self, list_):
        item = self._list_to_numpy(list_)
        item = self._numpy_to_tensor(item)
        return item


if __name__ == "__main__":
    rb = ReplayBuffer(capacity=100)
    # mrb = MongoReplayBuffer(capacity=100)

    n = 3

    observations = np.random.uniform(-1.0, 1.0, (32, 128))
    actions = np.random.randint(0, 7)
    next_observations = np.concatenate([observations[1:], np.random.uniform(-1.0, 1.0, (1, 128))])
    rewards = np.random.normal(0.0, 1.0)
    h_ins = [torch.randn(4, 1, 8) for _ in range(n)]
    # h_outs = [torch.randn(4, 1, 8) for _ in range(n)]
    done = np.random.choice([False, True])

    data = (observations, actions, next_observations, rewards, h_ins, done)
    rb.push(*data)

    data = rb.sample(1)
    print(data)
