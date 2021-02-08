#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import abc
import json
import random

import numpy as np
import pymongo

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)
    MONGO_USERNAME = config["mongo"]["username"]
    MONGO_PASSWORD = config["mongo"]["password"]


class ReplayMemory(abc.ABC):
    pass


class ReplayBuffer:

    def __init__(self, capacity=1_000_000, seed=0):
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

    def clear(self):
        self.buffer.clear()
        self.position = 0

    def __len__(self):
        return len(self.buffer)


class MongoReplayBuffer:

    def __init__(self, username=MONGO_USERNAME, password=MONGO_PASSWORD):
        client = pymongo.MongoClient(f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@localhost:27017/")
        database = client["rimpac"]
        self._collection = database["trajectory"]

    def push(self, state, action, reward, next_state, done):
        """
        Args:
            data: dict | list[dict]
            {
                "timestamp": datetime.datetime,
                ...
            }
        """
        data = {
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": reward,
            "next_state": next_state.tolist(),
            "done": done
        }
        _ = self._collection.insert_one(data)

    def sample(self, batch_size):
        pipeline = [
            {"$sample": {"size": batch_size}}
        ]
        batch = map(self._decode, self._collection.aggregate(pipeline))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def clear(self):
        self._collection.drop()

    def __len__(self):
        return self._collection.estimated_document_count()

    def _decode(self, row):
        return (
            np.array(row["state"]),
            np.array(row["action"]),
            np.array(row["reward"]),
            np.array(row["next_state"]),
            np.array(row["done"])
        )


if __name__ == "__main__":
    buffer = MongoReplayBuffer()
    buffer.clear()
    print(len(buffer))

    memory = []

    obs = np.random.uniform(-1.0, 1.0, (16, 4, 61))
    action = np.random.uniform(-1.0, 1.0, (1, 20))
    next_obs = np.random.uniform(-1.0, 1.0, (16, 4, 61))
    reward = np.random.uniform(-1.0, 1.0)
    done = False
    memory.append((obs, action, reward, next_obs, done))

    for s, a, r, s_, d in np.concatenate((memory,)):
        buffer.push(s, a, r, s_, d)

    print(len(buffer))

    s, a, r, s_, d = buffer.sample(1)
    print(f's: {s.shape}')
    print(f'a: {a.shape}')
    print(f'r: {r.shape}')
    print(f's_: {s_.shape}')
    print(f'd: {d.shape}')
