#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class ReplayBuffer:

    def __init__(
        self,
        max_steps,
        num_agents,
        observation_dims,
        action_dims
    ):
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.act_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []

        for _ in range(num_agents):
            self.obs_buffs.append(np.zeros((max_steps, observation_dims), dtype=np.float32))
            self.act_buffs.append(np.zeros((max_steps, action_dims), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_steps, observation_dims), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))

        self.filled_i = 0
        self.curr_i = 0

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        entries = observations.shape[0]


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True


def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False


if __name__ == "__main__":
    pass
