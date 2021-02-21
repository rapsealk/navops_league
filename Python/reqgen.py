#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import queue
from pprint import pprint

requirements = []

q = queue.Queue()
for dir_ in os.listdir(os.path.abspath(os.path.dirname(__file__))):
    q.put(os.path.abspath(dir_))

while not q.empty():
    try:
        dir_ = q.get(True, 0.05)
    except queue.Empty:
        break

    print('dir_:', dir_)
    if os.path.isdir(dir_):
        for subdir in os.listdir(dir_):
            q.put(os.path.join(dir_, subdir))
    elif dir_.endswith('requirements.txt'):
        print('requirements.txt!')
        with open(dir_) as f:
            requirements += f.readlines()


pprint(requirements)

with open(os.path.join(os.path.dirname(__file__), 'req.txt'), 'w') as f:
    f.writelines(requirements)
