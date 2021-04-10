#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import MultiHeadAttention

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        # SiLU
        # torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.constant_(m.bias, 0)
        # ReLU
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.01)


class GatedActorCritic(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=512,
        # rnn_num_layers=1
    ):
        super(GatedActorCritic, self).__init__()
        self._rnn_input_size = 512
        self._rnn_output_size = 256
        # self._rnn_num_layers = rnn_num_layers

        self.mh = MultiHeadAttention(input_size[-1])
        # self.rnn = nn.GRU(input_size[-1], hidden_size, num_layers=rnn_num_layers)
        self.rnn = nn.GRUCell(input_size[-1], hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.actor_h = nn.Linear(hidden_size, hidden_size)
        self.actor_h2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, output_size)
        # TODO: Hierarchical Action (https://arxiv.org/abs/1906.05862)
        self.critic_h = nn.Linear(hidden_size, hidden_size)
        self.critic_h2 = nn.Linear(hidden_size, hidden_size)
        self.critic = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)

    def forward(self, x, h_in):
        attn, attn_w = self.mh(x, x, x)
        # x, h_out = self.rnn(attn, h_in)
        h_out = self.rnn(attn, h_in)
        x = F.relu(self.linear(h_out))
        ah = F.relu(self.actor_h(x))
        ah = F.relu(self.actor_h2(ah))
        logit = self.actor(ah)
        vh = F.relu(self.critic_h(x))
        vh = F.relu(self.critic_h2(vh))
        value = self.critic(vh)
        return logit, value, h_out, attn_w

    def reset_hidden_state(self):
        # return torch.zeros([self._rnn_num_layers, 1, self._rnn_input_size], dtype=torch.float)
        return torch.zeros([1, self._rnn_input_size], dtype=torch.float)

    @property
    def rnn_input_size(self):
        return self._rnn_input_size

    @property
    def rnn_output_size(self):
        return self._rnn_output_size


def main():
    batch_size = 1
    input_shape = (24,)
    output_size = 10
    inputs = np.random.uniform(-1.0, 1.0, size=(batch_size, *input_shape))
    inputs = torch.from_numpy(inputs).float()

    model = GatedActorCritic(input_shape, output_size)
    hidden = model.reset_hidden_state()

    logits, value, hidden, attention = model(inputs, hidden)
    print('logits:', logits.shape)
    print('value:', value.shape)
    print('hidden:', hidden.shape)
    print('attention:', attention.shape)


if __name__ == "__main__":
    main()
