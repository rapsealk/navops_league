#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BooleanMaskLayer(nn.Module):

    def __init__(self, output_size):
        super(BooleanMaskLayer, self).__init__()
        self._output_size = output_size

    def forward(self, x: torch.Tensor):
        x = x.clone().detach().cpu().squeeze().numpy()
        # Steer: -3 ~ -7 (-3, +4)
        # Speed: -8 ~ -12 (+1, -2)
        if x.ndim == 1:
            mask = np.ones(self._output_size)
            if x[-3] == 1.0:
                mask[4] = 0     # float("-inf")
            elif x[-7] == 1.0:
                mask[3] = 0     # float("-inf")

            if x[-8] == 1.0:
                mask[1] = 0     # float("-inf")
            elif x[-12] == 1.0:
                mask[2] = 0     # float("-inf")
            # mask = torch.FloatTensor(mask)
            mask = torch.tensor(mask, requires_grad=False)
        elif x.ndim == 2:
            mask = np.ones((x.shape[0], self._output_size))
            mask[np.where(x[:, -3] == 1.0), 4] = 0      # float('-inf')
            mask[np.where(x[:, -7] == 1.0), 3] = 0      # float('-inf')
            mask[np.where(x[:, -8] == 1.0), 1] = 0      # float('-inf')
            mask[np.where(x[:, -12] == 1.0), 2] = 0     # float('-inf')
            mask = torch.tensor(mask, requires_grad=False).unsqueeze(1)

        return mask


def main():
    class Model(nn.Module):

        def __init__(self, input_size, output_size, hidden_size=64):
            super(Model, self).__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.head = nn.Linear(hidden_size, output_size)
            self.mask = BooleanMaskLayer(output_size)

        def forward(self, x):
            y = F.relu(self.linear1(x))
            y = F.relu(self.linear2(y))
            y = self.head(y)
            # Boolean mask before softmax
            y = y + self.mask(x)
            pi = F.softmax(y, dim=0)
            return pi, y

    n = 5
    onehot = np.zeros((n,))
    onehot[np.random.randint(0, n)] = 1
    inputs = np.concatenate([
        np.random.uniform(0.0, 1.0, (3,)),
        onehot,
        np.random.uniform(0.0, 1.0, (2,))
    ])
    inputs = torch.FloatTensor(inputs)
    print(f'Inputs: {inputs}')

    model = Model(inputs.shape[0], 4)
    policy, logits = model(inputs)
    print(f'Logits: {logits}')
    print(f'Policy: {policy} ({np.sum(policy.detach().numpy())})')


if __name__ == "__main__":
    main()
