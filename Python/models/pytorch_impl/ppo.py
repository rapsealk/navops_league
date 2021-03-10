#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .mask import BooleanMaskLayer


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


class IntrinsicCuriosityModule(nn.Module):

    def __init__(self, input_size, output_size, resnet_size=4):
        super(IntrinsicCuriosityModule, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._resnet_size = resnet_size
        self._feature_size = 256

        self.feature = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, self._feature_size)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(self._feature_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, self._output_size)
        )

        self.residual = [
            nn.Sequential(
                nn.Linear(self._output_size + self._feature_size, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            ).to(self._device)
            for _ in range(2 * self._resnet_size)
        ]

        self.forward_model1 = nn.Sequential(
            nn.Linear(self._output_size + self._feature_size, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        self.forward_model2 = nn.Sequential(
            nn.Linear(self._output_size + self._feature_size, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        self.apply(weights_init_)

    def forward(self, state, next_state, action):
        encoded_state = self.feature(state)
        encoded_next_state = self.feature(next_state)
        pred_action = self.inverse_model(torch.cat((encoded_state, encoded_next_state), dim=1))
        pred_next_state_feature_original = self.forward_model1(torch.cat((encoded_state, encoded_next_state), dim=1))

        for i in range(self._resnet_size):
            pred_next_state_feature = self.residual[i*2](torch.cat((pred_next_state_feature_original, action), dim=1))
            pred_next_state_feature_original = self.residual[i*2+1](torch.cat((pred_next_state_feature, action), dim=1)) + pred_next_state_feature_original

        pred_next_state_feature = self.forward_model2(torch.cat((pred_next_state_feature_original, action), dim=1))

        return encoded_next_state, pred_next_state_feature, pred_action


class PPOAgent:

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        learning_rate=0.00001,
        cuda=True
    ):
        self._input_size = input_size
        self._output_size = output_size

        if cuda:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device("cpu")

        self._model = LstmActorCriticModel(
            input_size,
            output_size,
            hidden_size
        ).to(self._device)
        # self._icm = IntrinsicCuriosityModule(input_size, output_size).to(self._device)
        # self._optim = optim.RMSprop(self._model.parameters())
        """
        self._optim = optim.Adam(
            list(self._model.parameters()) + list(self._icm.parameters()),
            lr=learning_rate)
        """
        self._optim = optim.Adam(self._model.parameters(), lr=learning_rate)

        self._gamma = 0.98
        self._lambda = 0.95
        self._epsilon_clip = 0.2
        self._k = 3
        self._eta = 0.01

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

    def get_intrinsic_curiosity(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self._device)
        next_state = torch.FloatTensor(next_state).to(self._device)
        action = torch.LongTensor(action).to(self._device)

        action_onehot = torch.FloatTensor(len(action), self._output_size).to(self._device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        next_state_feature, pred_next_state_feature, pred_action = self._icm(state, next_state, action_onehot)
        intrinsic_reward = self._eta * (next_state_feature - pred_next_state_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

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
        pi_losses = []
        value_losses = []

        time_horizon = 128
        while len(s) > 0:
            s_batch, s = s[:time_horizon], s[time_horizon:]
            a_batch, a = a[:time_horizon], a[time_horizon:]
            r_batch, r = r[:time_horizon], r[time_horizon:]
            s_batch_, s_ = s_[:time_horizon], s_[time_horizon:]
            a_prob_batch, a_prob = a_prob[:time_horizon], a_prob[time_horizon:]
            dones_batch, dones = dones[:time_horizon], dones[time_horizon:]
            for _ in range(self._k):
                v_ = self._model.value(s_batch_, hiddens[1]).squeeze(1)
                td_target = r_batch + self.gamma * v_ * dones_batch
                v = self._model.value(s_batch, hiddens[0]).squeeze(1)
                delta = (td_target - v).detach().cpu().numpy()

                # GAE: Generalized Advantage Estimation
                advantages = []
                advantage = 0.0
                for item in reversed(delta):
                    advantage = self.gamma * self.lambda_ * advantage + item[0]
                    advantages.append([advantage])
                advantages.reverse()

                pi, _ = self._model.get_policy(s_batch, hiddens[0])
                pi_a = pi.squeeze(1).gather(1, a_batch.type(torch.int64))
                ratio = torch.exp(torch.log(pi_a) - torch.log(a_prob_batch))  # a/b == exp(log(a) - log(b))

                surrogates = (ratio * advantage,
                              torch.clamp(ratio, 1-self._epsilon_clip, 1+self._epsilon_clip) * advantage)
                pi_loss = -torch.min(*surrogates)
                value_loss = 0.5 * F.smooth_l1_loss(v, td_target.detach())  # detach()
                loss = pi_loss + value_loss
                # loss = -torch.min(*surrogates) + 0.5 * F.smooth_l1_loss(v, td_target.detach())    # detach()

                losses.append(loss.mean().item())
                pi_losses.append(pi_loss.mean().item())
                value_losses.append(value_loss.mean().item())

                self._optim.zero_grad()
                loss.mean().backward(retain_graph=True)
                self._optim.step()

        return np.mean(losses), np.mean(pi_losses), np.mean(value_losses)

    def supervised_learning(self, s, a, r):
        s, a, r = convert_to_tensor(self._device, s, a, r)
        r = r.unsqueeze(1)
        h_in = (torch.zeros([1, 1, self.rnn_output_size], dtype=torch.float).to(self._device),
                torch.zeros([1, 1, self.rnn_output_size], dtype=torch.float).to(self._device))
        h_out = (torch.zeros([1, 1, self.rnn_output_size], dtype=torch.float).to(self._device),
                 torch.zeros([1, 1, self.rnn_output_size], dtype=torch.float).to(self._device))
        policy, value, h_out = self._model(s, h_in)
        policy = policy.squeeze(1)
        target = torch.ones(policy.shape[0]).type(torch.int64).to(self._device)
        policy_loss = F.cross_entropy(policy, target)

        value = value.squeeze(1)
        value_loss = F.smooth_l1_loss(value, r)

        loss = policy_loss + value_loss
        mean_loss = loss.mean().item()

        self._optim.zero_grad()
        loss.mean().backward()
        self._optim.step()

        return mean_loss

    def append(self, data):
        self._memory.append(data)

    def state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def save(self, path: str):
        pathlib.Path(os.path.abspath(os.path.dirname(path))).mkdir(parents=True, exist_ok=True)
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
