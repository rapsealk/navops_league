#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
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
        try:
            client = pymongo.MongoClient(f'mongodb://{username}:{password}@localhost:{port}/', serverSelectionTimeoutMS=1000)
        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise e
        database = client["navops"]
        self._collection = database[self._id]

    def push(self, *args, keys=('observation', 'action', 'next_observation', 'reward', 'h_in', 'done')):
        document = {key: self.to_pyobj(arg) for key, arg in zip(keys, args)}
        _ = self._collection.insert_one(document)

    def sample(self, batch_size, on_policy=False):
        pipeline = [
            {"$sample": {"size": batch_size}}
        ]
        batch = tuple(map(self.restore_pyobj, self._collection.aggregate(pipeline)))
        return batch

    def extend(self, items, keys=('observation', 'action', 'next_observation', 'reward', 'h_in', 'done')):
        documents = tuple({key: self.to_pyobj(value) for key, value in zip(keys, item)} for item in items)
        _ = self._collection.insert_many(documents)

    def clear(self):
        self._collection.drop()

    def __len__(self):
        return self._collection.estimated_document_count()

    def to_pyobj(self, arg):
        if type(arg) is np.ndarray:
            arg = arg.tolist()
        elif isinstance(arg, (tuple, list)) and type(arg[0]) is torch.Tensor:
            arg = tuple(value.numpy().tolist() for value in arg)
        return arg

    def restore_pyobj(self, document):
        return (
            np.array(document["observation"], dtype=np.float32),
            document["action"],
            np.array(document["next_observation"], dtype=np.float32),
            document["reward"],
            tuple(torch.from_numpy(np.array(h_in)).float() for h_in in document["h_in"]),
            document["done"]
        )

    @property
    def id(self):
        return self._id

    @property
    def items(self):
        return tuple(map(self.restore_pyobj, self._collection.find({})))


if __name__ == "__main__":
    rb = ReplayBuffer(capacity=100)
    mrb = MongoReplayBuffer()

    n = 3

    p_times = []
    mp_times = []

    for _ in range(180):
        observations = np.random.uniform(-1.0, 1.0, (2, 4))
        actions = np.random.randint(0, 7)
        next_observations = np.concatenate([observations[1:], np.random.uniform(-1.0, 1.0, (1, 4))])
        rewards = np.random.normal(0.0, 1.0)
        h_ins = [torch.randn(4, 1, 8) for _ in range(n)]
        # h_outs = [torch.randn(4, 1, 8) for _ in range(n)]
        done = random.choice([False, True])

        data = (observations, actions, next_observations, rewards, h_ins, done)

        time_ = time.time()
        rb.push(*data)
        p_times.append(time.time() - time_)

        time_ = time.time()
        mrb.push(*data)
        mp_times.append(time.time() - time_)

    print('len:', len(mrb))
    items = mrb.items
    mrb.extend(items)
    print('len:', len(mrb))

    mrb.clear()

    print('rb times:', rbt := np.mean(p_times))
    print('mrb times:', mrbt := np.mean(mp_times))
    print('mrb / rb times:', mrbt / rbt)
