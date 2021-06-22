#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import multiprocessing as mp

import gym
import gym_navops   # noqa: F401

from utils import get_free_port

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=4)
args = parser.parse_args()


def spawn_env_process(worker_id=0, port=9090):
    build_path = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOps', 'NavOps.exe')
    # build_path = os.path.join('/Users', 'rapsealk', 'Desktop', 'NavOps.app')
    try:
        env = gym.make('NavOpsMultiDiscrete-v0', build_path=build_path, port=port)
    except Exception as e:
        print('Exception:', e)
        return
    done = False

    while not done:
        s, r, done, info = env.step(env.action_space.sample())

    print(f'Worker Id: {worker_id} finished!')


def main():
    n = args.n
    base_port = 9090
    ports = [get_free_port(base_port+i) for i in range(n)]
    processes = [mp.Process(target=spawn_env_process, args=(i, ports[i])) for i in range(n)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    for process in processes:
        process.close()


if __name__ == "__main__":
    main()
