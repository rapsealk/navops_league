#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def convert_to_tensor(device, *args):
    return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))


class ValueNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        action_size,
        hidden_size=256
    ):
        super(ValueNetwork, self).__init__()
        self.linear11 = nn.Linear(input_size + action_size, hidden_size)
        self.linear12 = nn.Linear(hidden_size, hidden_size)
        self.linear13 = nn.Linear(hidden_size, 1)

        self.linear21 = nn.Linear(input_size + action_size, hidden_size)
        self.linear22 = nn.Linear(hidden_size, hidden_size)
        self.linear23 = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)

    def forward(self, x):
        x1 = F.relu(self.linear11(x))
        x1 = F.relu(self.linear12(x1))
        q1 = self.linear13(x1)
        x2 = F.relu(self.linear21(x))
        x2 = F.relu(self.linear22(x2))
        q2 = self.linear23(x2)
        return q1, q2


class LstmPolicyModel(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        num_layers=32,
        batch_size=128
    ):
        super(LstmPolicyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, output_size)
        self.log_std_linearn = nn.Linear(hidden_size, output_size)

        self.apply(weights_init_)

        self.hidden = (torch.randn((num_layers, batch_size, hidden_size)),
                       torch.randn((num_layers, batch_size, hidden_size)))

        self.action_scale = torch.FloatTensor(np.array((1.0 - -1.0,)) / 2)
        self.action_bias = torch.FloatTensor(np.array((1.0 + -1.0,)) / 2)

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linearn(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(LstmPolicyModel, self).to(device)


if __name__ == "__main__":
    input_size = 61
    action_size = 21
    batch_size = 128
    q = ValueNetwork(input_size=input_size, action_size=action_size)
    pi = LstmPolicyModel(
        input_size=input_size,
        output_size=action_size,
        batch_size=batch_size
    )

    inputs = np.random.normal(0, 1, (batch_size, 4, input_size))
    inputs = torch.FloatTensor(inputs)
    print('inputs.shape:', inputs.shape)
    actions, probs, means = pi.sample(inputs)
    print('actions.shape:', actions.shape)

    inputs = torch.cat([inputs, actions], dim=2)
    q1, q2 = q(inputs)
    print('values.shape:', q1.shape, q2.shape)
