#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import random
from collections import deque

import numpy as np


class EpisodeMemory:

    def __init__(self,
                 random_update=True,
                 max_epi_num=10000,
                 max_epi_len=200,
                 batch_size=1,
                 lookup_step=128):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit('It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        # -------------------- RANDOM UPDATE -------------------- #
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)

            # check_flag = True   # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:     # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=True, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1)     # sample buffer with minstep size
                    sample = episode.sample(random_update=True, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        # -------------------- SEQUENTIAL UPDATE -------------------- #
        else:   # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=False))

        return sampled_buffer, len(sampled_buffer[0][0])    # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, *args):
        self.obs.append(args[0])
        self.action.append(args[1])
        self.reward.append(args[2])
        self.next_obs.append(args[3])
        self.done.append(args[4])

    def sample(self, random_update=False, lookup_step=None, idx=None):
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return obs, action, reward, next_obs, done

    def __len__(self) -> int:
        return len(self.obs)


if __name__ == "__main__":
    pass
