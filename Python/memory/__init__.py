#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import abc
import json
import random
from collections import deque
from datetime import datetime
from uuid import uuid4

import numpy as np
import pymongo

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)
    MONGO_USERNAME = config["mongo"]["username"]
    MONGO_PASSWORD = config["mongo"]["password"]


class ReplayBuffer:

    def __init__(self, capacity=1_000_000, seed=0):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class MongoReplayBuffer:

    def __init__(self, username=MONGO_USERNAME, password=MONGO_PASSWORD):
        client = pymongo.MongoClient(f"mongodb://{username}:{password}@localhost:27017/")
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
        del batch
        return state, action, reward, next_state, done

    def clear(self):
        pass    # self._collection.drop()

    def __len__(self):
        return self._collection.estimated_document_count()

    def _decode(self, row):
        return (
            np.array(row["state"]),
            np.array(row["action"]),
            row["reward"],
            np.array(row["next_state"]),
            row["done"]
        )


class FirebaseReplayBuffer:
    pass    # TODO: raise NotImplementedError()


class MongoLocalMemory:

    def __init__(self, username=MONGO_USERNAME, password=MONGO_PASSWORD):
        self._id = str(uuid4()).replace('-', '')[:16]
        client = pymongo.MongoClient(f"mongodb://{username}:{password}@localhost:27017/")
        database = client["rimpac_tmp"]
        self._collection = database[self._id]

    # def __del__(self):
    #     print(f'MongoLocalMemory.__del__: collection({self._id}).drop()')
    #     self._collection.drop()
    #     print(11)

    def __iter__(self):
        for data in self._collection.find({}, no_cursor_timeout=True):
            yield self._decode(data)

    def append(self, data):
        """
        Args:
            data: dict | list[dict]
            {
                "timestamp": datetime.datetime,
                ...
            }
        """
        state, action, reward, next_state, done = data
        data = {
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": reward.tolist(),  # TODO
            "next_state": next_state.tolist(),
            "done": done
        }
        _ = self._collection.insert_one(data)

    def tolist(self):
        return tuple(map(self._decode, self._collection.find()))

    def clear(self):
        self._collection.drop()
        print(f'[{datetime.now().isoformat()}] MongoLocalMemory.clear: collection({self._id}).drop()')

    def _decode(self, row):
        return (
            np.array(row["state"]),
            np.array(row["action"]),
            row["reward"],
            np.array(row["next_state"]),
            row["done"]
        )

    @property
    def id(self):
        return self._id


if __name__ == "__main__":
    # buffer = MongoReplayBuffer()
    # buffer.clear()
    # print(len(buffer))

    memory = MongoLocalMemory()

    for _ in range(16):
        obs = np.random.uniform(-1.0, 1.0, (16, 4, 51))
        action = np.random.uniform(-1.0, 1.0, (1, 8))
        next_obs = np.random.uniform(-1.0, 1.0, (16, 4, 51))
        reward = np.random.uniform(-1.0, 1.0)
        done = False
        memory.append((obs, action, reward, next_obs, done))

    print(np.array(memory.tolist()).shape)
    for s, a, r, s_, d in np.concatenate((memory.tolist(),)):
        # print(s.shape, a.shape, r.shape, s_.shape, d.shape)
        print(s, a, r, s_, d)
    # print(memory.tolist()[-1])

    memory.clear()
    print(np.array(memory.tolist()).shape)
    """
    for s, a, r, s_, d in np.concatenate((memory.tolist(),)):
        # buffer.push(s, a, r, s_, d)
        print(s, a, r, s_, d)
    """

    """
    print(len(buffer))

    s, a, r, s_, d = buffer.sample(1)
    print(f's: {s.shape}')
    print(f'a: {a.shape}')
    print(f'r: {r.shape}')
    print(f's_: {s_.shape}')
    print(f'd: {d.shape}')
    """
