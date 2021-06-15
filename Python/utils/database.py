#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json
import time
from copy import copy

import numpy as np
import pymongo

from __init__ import generate_id

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

    def put_dict(self, dict_):
        return self.put(**dict_)

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


def main():
    pass


if __name__ == "__main__":
    main()
