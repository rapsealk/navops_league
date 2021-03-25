#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import json
import pprint
from datetime import datetime

# import pandas as pd
from pymongo import MongoClient
# import sqlalchemy as db

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)
    MONGO_ROOT_USERNAME = config["mongo"]["username"]
    MONGO_ROOT_PASSWORD = config["mongo"]["password"]


class MongoWrapper:
    pass


"""
def Dictable(cls):
    def decorated():
        cls.to_dict = lambda: cls.__dict__.copy()
        return cls
    return decorated
"""


"""
def main():
    engine = db.create_engine('mysql+pymysql://root:1111@localhost/navops')
    connection = engine.connect()
    metadata = db.MetaData()
    table = db.Table('test', metadata, autoload=True, autoload_with=engine)
    print('table:', table.columns.keys())
"""


def random_sample(collection, n=3):
    # { $sample: { size: <positive integer> } }
    pipeline = [
        {"$sample": {"size": n}}
    ]
    return collection.aggregate(pipeline)


def main():
    mongo = MongoClient("mongodb://{}:{}@localhost:27017/".format(MONGO_ROOT_USERNAME, MONGO_ROOT_PASSWORD))
    print(mongo.list_database_names())

    database = mongo["navops"]
    collection = database["transition"]

    x = collection.insert_one({
        "timestamp": datetime.now(),
        "position": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        },
        "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }
    })
    print(x.inserted_id)

    """
    x = collections.insert_many([
        { ... }
    ])
    print(x.inserted_ids)
    """

    sample = random_sample(collection)
    # print('sample:', list(sample))
    # print('sample:', )
    pprint.pprint(list(sample))


if __name__ == "__main__":
    main()
