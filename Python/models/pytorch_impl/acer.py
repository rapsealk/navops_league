#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .mask import BooleanMaskLayer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

    def __init__(self, input_size, output_size, hidden_size=256):
        super(LstmActorCriticModel, self).__init__()
        self._rnn_input_size = 256
        self._rnn_output_size = 128

        self.encoder = nn.Linear(input_size, self._rnn_input_size)
        self.rnn = nn.LSTM(self._rnn_input_size, self._rnn_output_size)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self._rnn_output_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.bn = nn.BatchNorm1d(hidden_size)
        self.actor_h = nn.Linear(hidden_size, hidden_size)
        self.actor_h2 = nn.Linear(hidden_size, hidden_size)
        # self.actor_h_bn = nn.BatchNorm1d(hidden_size)
        self.actor = nn.Linear(hidden_size, output_size)
        self.critic_h = nn.Linear(hidden_size, hidden_size)
        self.critic_h2 = nn.Linear(hidden_size, hidden_size)
        # self.critic_h_bn = nn.BatchNorm1d(hidden_size)
        self.critic = nn.Linear(hidden_size, 1)

        self.mask = BooleanMaskLayer(output_size)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.apply(weights_init_)

    def forward(self, x, hidden):
        x = F.silu(self.encoder(x))
        x = x.view(-1, 1, self._rnn_input_size)
        x, hidden = self.rnn(x, hidden)
        x = F.silu(self.linear(x))
        x = F.silu(self.linear2(x))
        p = F.silu(self.actor_h(x))
        p = F.silu(self.actor_h2(p))
        p = self.actor(p)
        v = F.silu(self.critic_h(x))
        v = F.silu(self.critic_h2(v))
        v = self.critic(v)
        return p, v, hidden

    def get_policy(self, inputs, hidden):
        x = F.silu(self.encoder(inputs))
        x = x.view(-1, 1, self._rnn_input_size)
        x, hidden = self.rnn(x, hidden)
        # x = self.flatten(x)
        x = F.silu(self.linear(x))
        x = F.silu(self.linear2(x))
        # x = F.silu(self.bn(self.linear(x)))
        x = F.silu(self.actor_h(x))
        x = F.silu(self.actor_h2(x))
        # x = F.silu(self.actor_h_bn(self.actor_h(x)))
        x = self.actor(x)
        x = x + self.mask(inputs).to(self._device)
        policy = F.softmax(x, dim=2)
        return policy, hidden

    def value(self, x, hidden):
        x = F.silu(self.encoder(x).unsqueeze(0))
        x = x.view(-1, 1, self._rnn_input_size)
        x, hidden = self.rnn(x, hidden)
        # x = F.silu(self.bn(self.linear(x)))
        x = F.silu(self.linear(x))
        x = F.silu(self.linear2(x))
        # x = F.silu(self.critic_h_bn(self.critic_h(x)))
        x = F.silu(self.critic_h(x))
        x = F.silu(self.critic_h2(x))
        value = self.critic(x)
        return value

    @property
    def rnn_output_size(self):
        return self._rnn_output_size

    def to(self, device):
        self.mask = self.mask.to(device)
        self._device = device
        return super(LstmActorCriticModel, self).to(device)


class AcerAgent:

    def __init__(
        self,
        model,
        buffer,
        c_sampling_ratio=1.0,
        learning_rate=0.0001,
        cuda=True
    ):
        if cuda:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device("cpu")

        self._model = model.to(self._device)
        self._optim = optim.Adam(self._model.parameters(), lr=learning_rate)

        self._gamma = 0.98
        """
        self._lambda = 0.95
        self._epsilon_clip = 0.2
        self._k = 3
        self._eta = 0.01
        """
        self._c_sampling_ratio = c_sampling_ratio

        self._buffer = buffer

    def get_action(self, state, hidden):
        state = torch.FloatTensor(state).to(self._device)
        hidden = tuple(h_in.to(self._device) for h_in in hidden)
        probs, hidden_out = self._model.get_policy(state, hidden)
        del state, hidden
        action = Categorical(probs).sample().item()
        probs = probs.detach().cpu().numpy().squeeze()
        hidden_out = tuple(h_out.detach().cpu() for h_out in hidden_out)
        return action, probs[action], hidden_out

    def train(self, batch_size=4, on_policy=False):
        s, a, r, s_, a_prob, h_in, h_out, dones, begins = \
            [], [], [], [], [], [], [], [], []

        # sample
        if on_policy:
            batches = [self._buffer[-1]]
        else:
            batches = self._buffer.sample(batch_size)
            # batch = reduce(lambda x, y: x + y, self._buffer.sample(batch_size), [])
        for batch in batches:
            for i, data in enumerate(batch):
                s.append(data[0])
                a.append(data[1])
                r.append(data[2])
                s_.append(data[3])
                a_prob.append(data[4])
                h_in.append(data[5])
                h_out.append(data[6])
                dones.append(data[7])
                begins.append(i == 0)

        s, a, r, s_, a_prob, dones, begins = \
            convert_to_tensor(self._device, s, a, r, s_, a_prob, dones, begins)
        print(f'[{datetime.now().isoformat()}] s.shape: {s.shape}')
        a = a.unsqueeze(1)
        r = r.unsqueeze(1)
        a_prob = a_prob.unsqueeze(1)
        dones = dones.unsqueeze(1)
        h_in, h_out = h_in[0], h_out[0]
        hiddens = [(h_in[0].detach().to(self._device), h_in[1].detach().to(self._device)),
                   (h_out[0].detach().to(self._device), h_out[1].detach().to(self_device))]

        losses = []
        pi_losses = []
        value_losses = []

        q = self._model.value(s, hiddens[0]).squeeze(1)
        q_a = q.gather(1, a.type(torch.int64))
        pi, _ = self._model.get_policy(s, hiddens[0]).squeeze(1)
        pi_a = pi.gather(1, a.type(torch.int64))
        v = (q * pi).sum(1).unsqueeze(1).detach()

        rho = pi.detach() / a_prob
        rho_a = rho.gather(1, a.type(torch.int64))
        rho_bar = rho_a.clamp(max=self._c_sampling_ratio)
        correction_coeff = (1 - self._c_sampling_ratio / rho).clamp(min=0)

        q_retrace = v[-1] * dones[-1]
        q_retraces = []
        for i in reversed(range(len(r))):
            q_retrace = r[i] + gamma * q_retrace
            q_retraces.append(q_retrace.item())
            q_retrace = rho_bar[i] * (q_retrace - q_a[i]) + v[i]

            if begins[i] and i != 0:
                q_retrace = v[i-1] * dones[i-1]     # when a new sequence begins

        q_retraces.reverse()
        q_retrace = torch.FloatTensor(q_retraces).unsqueeze(1)

        loss1 = -rho_bar * torch.log(pi_a) * (q_retrace - v)
        loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach() - v)  # bias correction term
        loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_retrace)

        loss_value = loss.mean().item()

        self._optim.zero_grad()
        loss.mean().backward()
        self._optim.step()

        return loss_value

    @property
    def buffer(self):
        return self._buffer


if __name__ == "__main__":
    pass
