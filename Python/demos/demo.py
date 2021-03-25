#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
from mlagents.trainers.buffer import BufferKey, ObservationKeyPrefix
from mlagents.trainers.demo_loader import demo_to_buffer

DEMO_PATH = os.path.join(os.path.dirname(__file__), 'NavOps.demo')


class DemoLoader:

    def load(self, path: str = DEMO_PATH):
        # assert obs.shape[-1] == 75
        brain_params, demo_buffer = demo_to_buffer(DEMO_PATH, sequence_length=1)
        observations = demo_buffer[ObservationKeyPrefix.OBSERVATION, 0]
        discrete_actions = demo_buffer[BufferKey.DISCRETE_ACTION]
        # continuous_actions = demo_buffer[BufferKey.CONTINUOUS_ACTION]
        # memory = demo_buffer[BufferKey.MEMORY]
        # print(f'Rewards: {rewards}, {type(rewards)}')
        rewards = demo_buffer[BufferKey.ENVIRONMENT_REWARDS]
        # rewards = np.array([r for r in rewards if r != 0])
        dones = demo_buffer[BufferKey.DONE]
        # return observations, discrete_actions, rewards, dones
        return np.array(observations), \
               np.array(discrete_actions), \
               np.array(rewards), \
               np.array(dones)


def decode_observation(obs):
    assert obs.shape[-1] == 75
    print('[Observation]')
    # print(f'Position(x, y): {obs[0:2]}')
    # print(f'Rotation(cos, sin): {obs[2:4]}')
    print(f'Target Position(x, y): {obs[4:6]}')
    # print(f'Target Rotation(cos, sin): {obs[6:8]}')
    print('Weapons..')
    for i in range(6):
        index = 8 + i * 7
        print(f'- [{i+1}] Rotation(x, cos, sin): {obs[index:index+3]}')
        # print(f'-     Reloaded: {obs[index+3]}')
        # print(f'-     Cooldown: {obs[index+4]}')
        # print(f'-     Damaged: {obs[index+5]}')
        # print(f'-     Repair Prog: {obs[index+6]}')
    print(f'Aiming Point(x): {obs[50:59]}')
    # print(f'Aiming Point(cos, sin): {obs[59:61]}')
    # print(f'Ammo: {obs[61]}')
    # print(f'Fuel: {obs[62]}')
    # print(f'Speed Level: {obs[63:68]}')
    # print(f'Steer Level: {obs[68:73]}')
    # print(f'Health: {obs[73]}')
    # print(f'Target Health: {obs[74]}')


def main():
    s, a, r, d = DemoLoader().load(DEMO_PATH)
    print(f'states: {s.shape}, {s.dtype}')
    print(f'actions: {a.shape}')
    print(f'rewards: {r.shape}')
    print(f'dones: {d.shape}')
    print(f'episode: {d[d != 0].shape}')


if __name__ == "__main__":
    main()
