#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.multiprocessing as mp  # noqa: F401
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from sac import SoftActorCriticAgent
from memory import ReplayBuffer


class MockBoxSpace:

    def __init__(self, shape: tuple):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def low(self):
        return np.zeros(self._shape)

    @property
    def high(self):
        return np.ones(self._shape)


class SharedGradientBuffer:

    def __init__(self, model: nn.Module):
        self._grads = {}
        for name, param in model.named_parameters():
            self._grads[f'{name}_grad'] = torch.ones(param.size()).share_memory_()

    def add_gradient(self, model: nn.Module):
        for name, param in model.named_parameters():
            self._grads[f'{name}_grad'] += param.grad.data

    def reset(self):
        for name, _ in self._grads.items:
            self._grads[name].fill_(0)

    @property
    def grads(self):
        return self._grads


if __name__ == "__main__":
    observation_space = np.zeros((4,))
    action_space = MockBoxSpace((2,))
    global_model = SoftActorCriticAgent(observation_space.shape[0], action_space)
    # global_model.share_memory()
    worker_model = SoftActorCriticAgent(observation_space.shape[0], action_space)
    worker_model.set_state_dict(global_model.get_state_dict())
    # shared_grad_buffer = SharedGradientBuffer(global_model.policy)
    # critic_shared_grad_buffer = SharedGradientBuffer(global_model.critic)
    buffer = ReplayBuffer(10000)

    writer = SummaryWriter('runs/{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    state = np.random.normal(-1.0, 1.0, (4,))
    for episode in range(1000):
        for _ in range(256):
            action = worker_model.select_action(state)
            next_state, reward, done = (np.random.normal(-1.0, 1.0, (4,)),
                                        np.random.normal(),
                                        False)
            buffer.push(state, action, reward, next_state, done)
            state = next_state

        qf_loss, policy_loss, alpha_loss = worker_model.compute_gradient(buffer, 128, 0)
        writer.add_scalar('Q Loss', qf_loss, episode)
        writer.add_scalar('PI Loss', policy_loss, episode)
        print(f'[{datetime.now().isoformat()}] Episode #{episode}: {qf_loss}, {policy_loss}')

        global_model.descent_gradient(worker_model, qf_loss, policy_loss, alpha_loss)
        worker_model.set_state_dict(global_model.get_state_dict())
