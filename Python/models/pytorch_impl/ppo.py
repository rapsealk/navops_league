#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


def weights_init_(m):
    if isinstance(m, nn.Linear):
        # ReLU
        # torch.nn.init.kaiming_normal_(m.weight, gain=1)
        # torch.nn.init.constant_(m.bias, 0.01)
        # SiLU
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def convert_to_tensor(device, *args):
    return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))


class LstmActorCriticModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=1024):
        super(LstmActorCriticModel, self).__init__()
        self._rnn_input_size = 128
        self._rnn_output_size = 64

        self.encoder = nn.Linear(input_size, self._rnn_input_size)
        self.rnn = nn.LSTM(self._rnn_input_size, self._rnn_output_size)
        self.linear = nn.Linear(self._rnn_output_size, hidden_size)
        self.actor = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)

    def get_policy(self, x, hidden):
        x = F.silu(self.encoder(x))
        x = x.view(-1, 1, self._rnn_input_size)
        x, hidden = self.rnn(x, hidden)
        x = F.silu(self.linear(x))
        policy = F.softmax(self.actor(x), dim=2)
        return policy, hidden

    def value(self, x, hidden):
        x = F.silu(self.encoder(x))
        x = x.view(-1, 1, self._rnn_input_size)
        x, hidden = self.rnn(x, hidden)
        x = F.silu(self.linear(x))
        value = self.critic(x)
        return value

    @property
    def rnn_output_size(self):
        return self._rnn_output_size


class PPOAgent:

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        learning_rate=0.00001
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = LstmActorCriticModel(
            input_size,
            output_size,
            hidden_size
        ).to(self._device)
        # self._optim = optim.RMSprop(self._model.parameters())
        self._optim = optim.Adam(self._model.parameters(), lr=learning_rate)

        self._gamma = 0.98
        self._lambda = 0.95
        self._epsilon_clip = 0.2
        self._k = 3

        self._memory = []

    def get_action(self, state, hidden):
        state = torch.FloatTensor(state).to(self._device)
        hidden = tuple(h_in.to(self._device) for h_in in hidden)
        probs, hidden_out = self._model.get_policy(state, hidden)
        del state, hidden
        action = Categorical(probs).sample().item()
        probs = probs.detach().cpu().numpy().squeeze()
        hidden_out = tuple(h_out.detach().cpu() for h_out in hidden_out)
        return action, probs[action], hidden_out

    def train(self):
        s, a, r, s_, a_prob, h_in, h_out, dones = [], [], [], [], [], [], [], []
        for data in self._memory:
            s.append(data[0])
            a.append(data[1])
            r.append(data[2])
            s_.append(data[3])
            a_prob.append(data[4])
            h_in.append(data[5])
            h_out.append(data[6])
            dones.append(data[7])
        self._memory.clear()

        s, a, r, s_, a_prob, dones = convert_to_tensor(self._device, s, a, r, s_, a_prob, dones)
        a = a.unsqueeze(1)
        r = r.unsqueeze(1)
        a_prob = a_prob.unsqueeze(1)
        dones = dones.unsqueeze(1)
        h_in, h_out = h_in[0], h_out[0]
        hiddens = [(h_in[0].detach().to(self._device), h_in[1].detach().to(self._device)),
                   (h_out[0].detach().to(self._device), h_out[1].detach().to(self._device))]

        losses = []

        for _ in range(self._k):
            v_ = self._model.value(s_, hiddens[1]).squeeze(1)
            td_target = r + self.gamma * v_ * dones
            v = self._model.value(s, hiddens[0]).squeeze(1)
            delta = (td_target - v).detach().cpu().numpy()

            # GAE: Generalized Advantage Estimation
            advantages = []
            advantage = 0.0
            for item in reversed(delta):
                advantage = self.gamma * self.lambda_ * advantage + item[0]
                advantages.append([advantage])
            advantages.reverse()

            pi, _ = self._model.get_policy(s, hiddens[0])
            pi_a = pi.squeeze(1).gather(1, a.type(torch.int64))
            ratio = torch.exp(torch.log(pi_a) - torch.log(a_prob))  # a/b == exp(log(a) - log(b))

            surrogates = (ratio * advantage,
                          torch.clamp(ratio, 1-self._epsilon_clip, 1+self._epsilon_clip) * advantage)
            loss = -torch.min(*surrogates) + 0.5 * F.smooth_l1_loss(v, td_target.detach())    # detach()

            losses.append(loss.mean().item())

            self._optim.zero_grad()
            loss.mean().backward(retain_graph=True)
            self._optim.step()

        return np.mean(losses)

    def append(self, data):
        self._memory.append(data)

    def state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        torch.save({
            "params": self._model.state_dict(),
            # "optim": self._optim.parameters(),
            # TODO: epsilon
        }, path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self._model.load_state_dict(state_dict["params"])
        # self._optim.load(state_dict["optim"])

    @property
    def gamma(self):
        return self._gamma

    @property
    def lambda_(self):
        return self._lambda

    @property
    def rnn_output_size(self):
        return self._model.rnn_output_size


if __name__ == "__main__":
    pass
