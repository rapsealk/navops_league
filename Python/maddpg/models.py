#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot(items, max_range=10):
    # print(f'[ONEHOT] items: {items} ({items.shape})')
    # print(f'[ONEHOT] max_range: {max_range}')
    x = np.zeros((items.shape[0], max_range))
    # print(f'[ONEHOT] x: {x} ({x.shape})')
    """
    x = torch.zeros((items.shape[0], max_range))   # .astype(torch.bool)
    for i, index in enumerate(items):
        print(f'[ONEHOT] - (i, idx): {i}, {index} (x[i]: {x[i].shape}, {x[i, index].shape})')
        x[i][index] = 1
    print(f'[ONEHOT] x: {x} ({x.shape})')
    # x[range(items.shape[0]), items] = 1
    return x
    """
    x[range(items.shape[0]), items] = 1
    # print(f'[ONEHOT] x: {x} ({x.shape})')
    return torch.from_numpy(x).view(-1) # .bool()


class Actor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.action_head(x)
        prob = F.softmax(x, dim=-1)
        return prob


class Critic(nn.Module):

    def __init__(self, input_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self._action_size = action_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.q_value_head = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # state = torch.cat(state, dim=1)
        state = state.view(state.shape[0], -1)
        # for i in range(len(action)):
        #     action[i] /= self.max_action
        # action = torch.cat(action, dim=1)

        # print(f'[CRITIC] action: {action} ({action.shape})')
        action = torch.transpose(action.byte(), dim0=1, dim1=0)
        # print(f'[CRITIC] action.transpose: {action} ({action.shape})')
        oh_action = []
        for a in action:
            oh_action.append(onehot(a.numpy(), max_range=self.action_size))
        action = torch.stack(oh_action)
        # print(f'[CRITIC] action.onehot: {action} ({action.shape})')

        x = torch.cat([state, action], dim=-1).float()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        q_value = self.q_value_head(x)
        return q_value

    @property
    def action_size(self):
        return self._action_size


def main():
    x = np.random.randint(0, 10, (3,))
    oh = onehot(x)
    print(oh)


if __name__ == "__main__":
    main()
