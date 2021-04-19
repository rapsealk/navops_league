#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def onehot(items, max_range=10):
    x = np.zeros((items.shape[0], max_range))
    x[range(items.shape[0]), items] = 1
    return torch.from_numpy(x).view(-1)


class Actor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64, rnn_hidden_size=64, batch_size=8):
        super(Actor, self).__init__()
        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size
        self._batch_size = batch_size

        self.rnn = nn.GRUCell(input_size, rnn_hidden_size)
        self.linear1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, output_size)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, h_in):
        # print(f'[ACTOR] forward(x={x.shape}, h_in={h_in.shape})')
        x = x.to(self.device)
        h_in = h_in.to(self.device)
        h_out = self.rnn(x, h_in)
        x = F.relu(self.linear1(h_out))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.action_head(x)
        prob = F.softmax(x, dim=-1)
        return prob, h_out

    def to(self, device):
        self._device = device
        return super(Actor, self).to(device)

    def reset_hidden_state(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return torch.zeros((batch_size, self.rnn_hidden_size))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device


class Critic(nn.Module):

    def __init__(self, input_size, action_size, hidden_size=64, rnn_hidden_size=64, batch_size=8):
        super(Critic, self).__init__()
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size
        self._batch_size = batch_size

        self.rnn = nn.GRUCell(input_size, rnn_hidden_size)
        self.linear1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.q_value_head = nn.Linear(hidden_size, 1)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state, action, h_in):
        # state = torch.cat(state, dim=1)
        state = state.view(state.shape[0], -1).to(self.device)
        # for i in range(len(action)):
        #     action[i] /= self.max_action
        # action = torch.cat(action, dim=1)

        action = torch.transpose(action.byte(), dim0=1, dim1=0)
        oh_action = []
        for a in action:
            oh_action.append(onehot(a.cpu().numpy(), max_range=self.action_size))
        action = torch.stack(oh_action).to(self.device)

        x = torch.cat([state, action], dim=-1).float().to(self.device)
        h_in = h_in.to(self.device)
        h_out = self.rnn(x, h_in)
        x = F.relu(self.linear1(h_out))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        q_value = self.q_value_head(x)
        return q_value, h_out

    def to(self, device):
        self._device = device
        return super(Critic, self).to(device)

    def reset_hidden_state(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return torch.zeros((batch_size, self.rnn_hidden_size))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def device(self):
        return self._device


def main():
    rollout = 4
    action_size = 10
    inputs = np.random.uniform(-1.0, 1.0, (rollout, 24))
    pi = Actor(input_size=inputs.shape[-1], output_size=action_size, hidden_size=32, rnn_hidden_size=64, batch_size=rollout)
    h_in = pi.reset_hidden_state()

    inputs = torch.from_numpy(inputs).float()
    print('inputs:', inputs.shape)
    logits, h_out = pi(inputs, h_in)
    print('logits:', logits.shape)
    print('h_in:', h_in.shape, 'h_out:', h_out.shape)

    action = Categorical(logits=logits[-1]).sample().numpy()
    print('action:', action)

    print('---')

    critic = Critic(inputs.shape[-1] * rollout + action_size, action_size, hidden_size=32, rnn_hidden_size=64, batch_size=1)
    critic = critic.to(critic.device)
    h_in = critic.reset_hidden_state(batch_size=1)

    inputs = inputs.unsqueeze(0)
    actions = torch.from_numpy(action).unsqueeze(0).unsqueeze(0)
    q, h_out = critic(inputs, actions, h_in)
    print('q:', q)
    print('h_in:', h_in.shape, 'h_out:', h_out.shape)


if __name__ == "__main__":
    main()
