#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pickle
from collections import deque

import numpy as np

from memory import MongoLocalMemory


def main():
    tuple_ = (1200, 1200)
    pickle.dumps(tuple_)

    deque_ = deque(maxlen=100)
    pickle.dumps(deque_)

    numpy_ = np.zeros((100, 100, 100))
    pickle.dumps(numpy_)

    memory = []
    pickle.dumps(memory)

    memory = MongoLocalMemory()
    memory.append((np.zeros((1,)), np.zeros((1,)), np.ones((1,)), np.ones((1,)), False))
    print('MongoLocalMemory:', memory)
    print('MongoLocalMemory.tolist():', memory.tolist())
    pickle.dumps(memory.id)
    memory.clear()
    pickle.dumps(memory._collection)
    pickle.dumps(memory)


if __name__ == "__main__":
    main()
