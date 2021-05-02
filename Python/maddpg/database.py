#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json

import pymongo

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json'), 'r') as f:
    config = ''.join(f.readlines())
    config = json.loads(config)
    MONGO_USERNAME = config["mongo"]["username"]
    MONGO_PASSWORD = config["mongo"]["password"]


class MongoDatabase:

    def __init__(self):
        self._id = generate_id()
        # mongodb://root:1111@localhost:27017/
        client = pymongo.MongoClient(f'mongodb://{username}:{password}@localhost:{port}/')
        database = client["navops"]
        self._collection = database["train"]


if __name__ == "__main__":
    pass
