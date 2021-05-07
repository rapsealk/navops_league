#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        rnn_hidden_size,
        action_size
    ):
        super(TimeSeriesNetwork, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size

        self.linear1 = nn.Linear(input_size, rnn_hidden_size)
        self.rnn = nn.GRUCell(rnn_hidden_size, rnn_hidden_size)
        self.linear2 = nn.Linear(rnn_hidden_size, action_size)

    def forward(self, obs, hidden_state):
        x = F.relu(self.linear1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_size)
        h_out = self.rnn(x, h_in)
        q = self.linear2(h_out)
        return q, h_out

    def reset_hidden_state(self):
        return torch.zeros(1, self.rnn_hidden_size)

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size


class QMixingNetwork(nn.Module):

    def __init__(
        self,
        state_size,
        hyper_hidden_size,
        qmix_hidden_size,
        n=3
    ):
        super(QMixingNetwork, self).__init__()
        # self.args = args
        self._state_size = state_size
        self._qmix_hidden_size = qmix_hidden_size
        self._n = n

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_size, hyper_hidden_size),
            nn.ReLU(),
            nn.Linear(hyper_hidden_size, n * qmix_hidden_size)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_size, hyper_hidden_size),
            nn.ReLU(),
            nn.Linear(hyper_hidden_size, qmix_hidden_size)
        )

        self.hyper_b1 = nn.Linear(state_size, qmix_hidden_size)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_size, qmix_hidden_size),
            nn.ReLU(),
            nn.Linear(qmix_hidden_size, 1)
        )

    def forward(self, state, q_value):
        episode_num = q_value.shape[0]
        q_value = q_value.view(-1, 1, self.n)
        state = state.reshape(-1, self.state_size)

        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)

        w1 = w1.view(-1, self.n, self.qmix_hidden_size)
        b1 = b1.view(-1, 1, self.qmix_hidden_size)

        hidden = F.elu(torch.bmm(q_value, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state))
        b2 = self.hyper_b2(state)

        w2 = w2.view(-1, self.qmix_hidden_size, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)

        return q_total

    @property
    def state_size(self):
        return self._state_size

    @property
    def qmix_hidden_size(self):
        return self._qmix_hidden_size

    @property
    def n(self):
        return self._n


if __name__ == "__main__":
    pass
