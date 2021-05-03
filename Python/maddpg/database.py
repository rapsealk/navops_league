#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
from copy import copy

import numpy as np
import pymongo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import generate_id

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json'), 'r') as f:
    config = ''.join(f.readlines())
    config = json.loads(config)
    MONGO_USERNAME = config["mongo"]["username"]
    MONGO_PASSWORD = config["mongo"]["password"]


class MongoDatabase:

    def __init__(self, username=MONGO_USERNAME, password=MONGO_PASSWORD, port=27017):
        self._id = generate_id()
        # mongodb://root:1111@localhost:27017/
        client = pymongo.MongoClient(f'mongodb://{username}:{password}@localhost:{port}/')
        database = client["navops"]
        self._collection = database["experience"]

    def ref(self, key):
        # return MongoDatabaseRef(self._collection[key])
        instance = copy(self)
        instance._collection = self._collection[key]
        return instance

    def put(self, **kwargs):
        """
        Args:
            value (dict): {"timestamp": 0}
        """
        value = {
            "timestamp": time.time(),
            **kwargs
        }
        value = self._encode(**value)
        result = self._collection.insert_one(value)
        return result.inserted_id

    def get(self, key=None):
        result = self._collection.find({})
        return tuple(result)

    def clear(self):
        self._collection.drop()

    def _encode(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, (np.float32, np.float64)):
                kwargs[key] = float(value)
            elif type(value) is np.ndarray:
                kwargs[key] = value.tolist()
            elif isinstance(value, (tuple, list)) and type(value[0]) is np.ndarray:
                kwargs[key] = tuple(val.tolist() for val in value)
            if isinstance(value, (tuple, list)) and isinstance(value[0], (np.float32, np.float64)):
                kwargs[key] = tuple(float(val) for val in value)
            elif type(value) is dict:
                kwargs[key] = self._encode(**value)
        return kwargs

    def __len__(self):
        return self._collection.estimated_document_count()

    @property
    def id(self):
        return self._id


"""
class MongoDatabaseRef:

    def __init__(self, collection):
        self._collection = collection

    def ref(self, key):
        return MongoDatabaseRef(self._collection[key])

    def put(self, **kwargs):
        value = {
            "timestamp": time.time(),
            **kwargs
        }
        return self._collection.insert_one(value)

    def get(self, key=None):
        result = self._collection.find({})
        return tuple(result)

    def clear(self):
        self._collection.drop()

    def __len__(self):
        return self._collection.estimated_document_count()
"""


def main():
    db = MongoDatabase()
    session_id = generate_id()
    print('session:', session_id)
    r = db.ref(session_id)
    print('ref:', r)

    print('db:', id(db), db.id, db._collection)
    print('rf:', id(r), r.id, r._collection)
    return

    value = {
        "position": {
            "x": 1.0, "y": 2.0, "z": 3.0
        }
    }
    result = r.put(**value)
    print('result:', result.inserted_id)

    print('get:', r.get())

    print('len:', len(r), end=' -> ')
    r.clear()
    print(len(r))


if __name__ == "__main__":
    main()
