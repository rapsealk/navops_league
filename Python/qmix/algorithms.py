#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import TimeSeriesNetwork, QMixingNetwork


def convert_to_tensor(device, *args):
    # return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))
    return torch.stack(tuple(map(lambda tensor: tensor.float().to(device), map(torch.from_numpy,
        map(lambda arr: np.array(arr, dtype=np.float32), args)
    ))))


class QMIX:

    def __init__(
        self,
        state_size,
        observation_size,
        action_size,
        rnn_hidden_size=64,
        hyper_hidden_size=64,
        qmix_hidden_size=32,
        # qtran_hidden_size=64,
        n=3,
        learning_rate=5e-4
    ):
        self._state_size = state_size
        self._observation_size = observation_size
        self._action_size = action_size
        self._rnn_hidden_size = rnn_hidden_size
        self._n = n

        input_shape = self._observation_size
        # if self.args.last_action
        input_shape += self._action_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        """

        self.rnn = TimeSeriesNetwork(input_shape, rnn_hidden_size, action_size)
        self.target_rnn = TimeSeriesNetwork(input_shape, rnn_hidden_size, action_size)
        self.qmix_net = QMixingNetwork(state_size, hyper_hidden_size, qmix_hidden_size, n=n)
        self.target_qmix_net = QMixingNetwork(state_size, hyper_hidden_size, qmix_hidden_size, n=n)

        self.rnn = self.rnn.to(device)
        self.target_rnn = self.target_rnn.to(device)
        self.qmix_net = self.qmix_net.to(device)
        self.target_qmix_net = self.target_qmix_net.to(device)

        self.target_rnn.load_state_dict(self.rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())

        parameters = list(self.qmix_net.parameters()) + list(self.rnn.parameters())
        self.optim = optim.RMSprop(parameters, lr=learning_rate)

        self.train_step = 0

    def learn(self, transitions, max_episode_len, epsilon=None):
        # episode_num = observations.shape[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # for agent in other_agents:
        #     agent.policy.to(device)

        batch_size = len(transitions)

        states, actions, next_states, rewards, h_ins, dones = [], [], [], [], [], []
        # avail_action, avail_action_next
        for s, a, s_, r, h_in, d in transitions:
            states.append(s)
            actions.append(a)
            next_states.append(s_)
            rewards.append(r)
            h_ins.append(h_in)
            dones.append(d)

        q_estimates, q_targets = self.get_q_value(transitions, max_episode_len)

        self.train_step += 1

    def get_q_value(self, transitions, max_episode_len):
        episode_num = len(transitions)
        q_est, q_target = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs

    def _get_inputs(self, obs, next_obs, u_onehot, transition_idx):
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []

        # if self.args.last_action:

    @property
    def action_size(self):
        return self._action_size

    @property
    def n(self):
        return self._n


def main():
    pass


if __name__ == "__main__":
    main()
