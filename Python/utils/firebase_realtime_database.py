#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
# import json
import random
import uuid

import firebase_admin
from firebase_admin import db
from firebase_admin import credentials


def first_or_null(func, iterables):
    return items[0] if (items := tuple(filter(func, iterables))) else None


def main():
    files = os.listdir(os.path.abspath(os.path.dirname(__file__)))
    if firebase_service_key := first_or_null(lambda f: f.endswith('.json'), files):
        cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), firebase_service_key))
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://distributedrl.firebaseio.com/"
        })
    else:
        return

    ref = db.reference('server/saving-data/fireblog')
    session_id = str(uuid.uuid4()).replace('-', '')
    episode_ref = ref.child(f'session/{session_id}')
    t = time.time()
    for _ in range(100):
        message = str(_)
        message = {
            "timestamp": time.time(),
            "message": message,
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
        episode_ref.set({
            f'{_}': message
        })
    print('Time:', time.time() - t)


if __name__ == "__main__":
    main()
