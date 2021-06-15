#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import random
import uuid
import time

from pikamon import Publisher, Subscriber
from database import MongoDatabase

parser = argparse.ArgumentParser()
# parser.add_argument('--session', default='abc123', type=str)
parser.add_argument('-s', '--subscribe', action='store_true')
args = parser.parse_args()


def random_message():
    return {
        "hp": random.random(),
        "position": [
            (random.random() - 0.5) * 2,
            (random.random() - 0.5) * 2
        ],
        "rotation": [
            (random.random() - 0.5) * 2,
            (random.random() - 0.5) * 2
        ],
        "opponent": {
            "hp": random.random(),
            "position": [
                (random.random() - 0.5) * 2,
                (random.random() - 0.5) * 2
            ],
            "rotation": [
                (random.random() - 0.5) * 2,
                (random.random() - 0.5) * 2
            ]
        },
        "action": random.randint(0, 7),
        "reward": (random.random() - 0.5) * 2
    }


def main():
    if args.subscribe:
        session_id = str(uuid.uuid4()).replace('-', '')[:16]
        database = MongoDatabase()
        ref = database.ref(session_id)
        sub = Subscriber()
        sub.add_pipeline(ref.put_dict)
        print('[Pikamon] subscribing..')
        try:
            sub.run()
        except KeyboardInterrupt:
            print('KeyboardInterrupted!')
    else:
        pub = Publisher()
        while input('continue? (y/n): ') == 'y':
            t = time.time()
            for _ in range(1000):
                pub.enqueue(random_message())
            print(f'Time: {time.time() - t}s')


if __name__ == "__main__":
    main()
