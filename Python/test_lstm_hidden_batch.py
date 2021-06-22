#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_num_layers=1):
        super(Model, self).__init__()
        self._hidden_size = hidden_size
        self._rnn_num_layers = rnn_num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=rnn_num_layers, batch_first=True)

    def forward(self, x, h_in):
        return self.lstm(x, h_in)

    def reset_hidden_state(self, batch_size=1):
        return (torch.zeros((self._rnn_num_layers, batch_size, self.hidden_size)),
                torch.zeros((self._rnn_num_layers, batch_size, self.hidden_size)))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def rnn_num_layers(self):
        return self._rnn_num_layers


def main():
    input_size = 8
    output_size = 2
    batch_size = 4
    m = Model(input_size, output_size)
    h_in = m.reset_hidden_state(batch_size)

    x = np.random.uniform(-1.0, 1.0, (batch_size, input_size))
    x = torch.from_numpy(x).float().unsqueeze(1)
    print('x, h_in:', x.shape, h_in[0].shape)

    y, h_out = m(x, h_in)
    print('y, h_out:', y.shape, h_out[0].shape)

    h_outs = []
    ys = []
    x_1 = x[0].unsqueeze(0)
    h_in_1 = m.reset_hidden_state(batch_size=1)
    y_1, h_out_1 = m(x_1, h_in_1)
    # print('x_, h_in_:', x_.shape, h_in_[0].shape)

    x_2 = x[1].unsqueeze(0)
    h_in_2 = m.reset_hidden_state(batch_size=1)
    y_2, h_out_2 = m(x_2, h_in_2)
    # print('x_, h_in_:', x_.shape, h_in_[0].shape)

    print('h_out[0]:', h_out[0].detach().numpy())
    print('h_out[1]:', h_out[1].detach().numpy())
    print('h_out_1:', tuple(map(lambda x: x.detach().numpy(), h_out_1)))
    print('h_out_2:', tuple(map(lambda x: x.detach().numpy(), h_out_2)))

    """
    for x_ in x:
        y_, h_in_ = m(x_.unsqueeze(0), h_in_)
        ys.append(y_)
        h_outs.append(h_in_)
    for h_out, h_out_ in zip(h_out, h_outs):
        # print(h_out, h_out_)
        h_out_ = torch.from_numpy(np.array([h_out_[0].detach(), h_out_[1].detach()]), dtype=np.float32)
        print(type(h_out), type(h_out_))
    """


if __name__ == "__main__":
    main()
