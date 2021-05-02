#!/usr/bin/python3
# -*- coding: utf-8 -*-
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(m, activation_fn=F.silu):
    if isinstance(m, nn.Linear):
        # ReLU
        if activation_fn is F.relu:
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)
        # SiLU
        elif activation_fn is F.silu:
            torch.nn.init.xavier_normal_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            """
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
            """
            torch.nn.init.constant_(param.data, 0.01)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            """
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
            """
            torch.nn.init.constant_(param.data, 0.01)


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RnnModel, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.apply(weights_init_)

    def forward(self, x, h_in):
        # print(f'Rnn.forward(x={x.shape}, h_in={h_in.shape})')
        x, h_out = self.rnn(x, h_in)
        return x, h_out

    def reset_hidden_state(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers


class RnnCellModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RnnCellModel, self).__init__()
        self._hidden_size = hidden_size
        self.rnn = nn.GRUCell(input_size, hidden_size)

        self.apply(weights_init_)

    def forward(self, x, h_in):
        # print(f'RnnCell.forward(x={x.shape}, h_in={h_in.shape})')
        h_out = self.rnn(x, h_in)
        return h_out

    def reset_hidden_state(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)

    @property
    def hidden_size(self):
        return self._hidden_size


class RnnTestCase(unittest.TestCase):

    def test_(self):
        pass


def main():
    batch_size = 4
    input_size = 8
    hidden_size = 16
    # inputs = np.random.uniform(-1.0, 1.0, (batch_size, 1, input_size))
    # inputs = torch.from_numpy(inputs).float()
    inputs = torch.randn(batch_size, 1, input_size)
    print('inputs:', inputs.shape, inputs)
    #return

    rnnmod = RnnModel(input_size, hidden_size)
    rnn_h_i = rnnmod.reset_hidden_state(batch_size)
    print('rnn_h_i:', rnn_h_i.shape, rnn_h_i)
    rnn_o, rnn_h_o = rnnmod(inputs, rnn_h_i)
    print('rnn_h_o:', rnn_h_o.shape, rnn_h_o)

    rnncmod = RnnCellModel(input_size, hidden_size)

    # 1.
    rnnc_h = rnncmod.reset_hidden_state(batch_size=1)
    # for input_ in inputs:
    for i in range(inputs.shape[0]):
        # print('input_:', input_.shape, input_)
        rnnc_h = rnncmod(inputs[i], rnnc_h)
        print('rnnc_h1:', rnnc_h.shape, rnnc_h)
    # print('rnnc_h1:', rnnc_h.shape, rnnc_h)

    """
    # 2.
    rnnc_h = rnncmod.reset_hidden_state(batch_size=4)
    rnnc_h = rnncmod(inputs.squeeze(), rnnc_h)
    print('rnnc_h2:', rnnc_h.shape, rnnc_h)
    """

    # print(f'rnn_h_o: {rnn_h_o.shape}')
    # print(rnn_h_o[0, 0])

    # print(f'rnnc_h: {rnnc_h.shape}')
    # print(rnnc_h)


if __name__ == "__main__":
    main()
    # unittest.main()
