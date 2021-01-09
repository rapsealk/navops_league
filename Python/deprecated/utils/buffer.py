#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import random

import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "firebase.json"))
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://distributedrl.firebaseio.com"
})


class ReplayBuffer:

    def __init__(self, capacity=1000000):
        self.memory = []
        self.capacity = capacity

    def append(self, data):
        self.memory.append(data)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, n):
        return random.sample(self.memory, n)


class FirebaseDatabase:

    def __init__(self):
        pass

    def ref(self, path: str) -> db.Reference:
        return db.reference(path)

    def get(self, path: str):
        return db.reference(path).get()

    def set(self, path: str, value: dict):
        db.reference(path).set(value)

    def list(self, path: str):
        #try:
        return db.reference(path).get()
        #except AttributeError:
        #    return []


class FirebaseReplayBuffer:

    def __init__(self):
        self.firebase = FirebaseDatabase()
        self.n = len(self.firebase.list("buffer"))

    def append(self, data):
        self.firebase.set("buffer/" + str(self.n + 1), data)
        self.n += 1


if __name__ == "__main__":
    buffer = FirebaseReplayBuffer()
    print(buffer.n)
    buffer.append({"hi": "world"})
    print(buffer.n)
    print(buffer.firebase.list("buffer"))
