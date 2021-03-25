#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import os
import json
from time import perf_counter
from datetime import datetime

import pymongo

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)
    MONGO_ROOT_USERNAME = config["mongo"]["username"]
    MONGO_ROOT_PASSWORD = config["mongo"]["password"]


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.database = pymongo.MongoClient("mongodb://{}:{}@localhost:27017/".format(MONGO_ROOT_USERNAME, MONGO_ROOT_PASSWORD))
        self.collection = self.database["navops"]["transition"]

        self.collection.drop()
        documents = []
        for _ in range(1_000_000):    # 240mb
            documents.append({
                "timestamp": datetime.now(),
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}
            })  # 240 Bytes
        self.collection.insert_many(documents)

    def test_random_sample_performance(self):
        size = 2048
        counter = perf_counter()
        pipeline = [
            {"$sample": {"size": size}}
        ]
        samples = self.collection.aggregate(pipeline)
        performance = perf_counter() - counter
        print(performance)
        self.assertEqual(len(list(samples)), size)
        self.assertLess(performance, 1.0)

    def tearDown(self):
        self.collection.drop()


if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromModule(TestDatabase)
    # unittest.TextTestRunner(verbosity=2).run(suite)
