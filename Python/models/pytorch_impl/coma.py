#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# epsilon = 0.5
# epsilon_annealing = 0.02
# epsilon_step = 750


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


def convert_to_tensor(device, *args):
    return map(lambda tensor: tensor.float().to(device), map(torch.tensor, args))   # noqa: E501


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Actor(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        rnn_hidden_size=64,
        rnn='gru'
    ):
        super(Actor, self).__init__()

        output_size = np.sum(output_size)

        assert rnn.lower() in ('lstm', 'gru')
        self.encoder = nn.Linear(input_size, hidden_size)
        if rnn.lower() == 'lstm':
            self.rnn = nn.LSTMCell(hidden_size, rnn_hidden_size)
        elif rnn.lower() == 'gru':
            self.rnn = nn.GRUCell(hidden_size, rnn_hidden_size)
        self.head = nn.Linear(rnn_hidden_size, output_size)

        self.apply(weights_init_)

        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size

    def reset_hidden_states(self):
        return self.encoder.weight.new(1, self.rnn_hidden_size).zero_()

    def forward(self, x, h_in):
        x = F.silu(self.encoder(x))
        x = x.view(-1, x.shape[-1])
        h_out = self.rnn(x, h_in)
        x = F.softmax(self.head(h_out), dim=1)  # * (1 - epsilon)
        return x, h_out

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size


class Critic(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128, n=3):
        super(Critic, self).__init__()

        output_size = np.sum(output_size)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size + hidden_size)
        self.linear3 = nn.Linear(hidden_size + hidden_size, output_size ** n)

        self.apply(weights_init_)

        self._hidden_size = hidden_size

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        x = self.linear3(x)
        return x

    @property
    def hidden_size(self):
        return self._hidden_size


class COMAAgent:

    def __init__(self, input_size, output_size, learning_rate=3e-4):
        self.actor = Actor(input_size, output_size)
        self._actor_optim = optim.Adam(self.actor.parameters(), lr=learning_rate)   # noqa: E501

    def get_action(self, state, hidden_in):
        logit, hidden_out = self.actor(state, hidden_in)
        action = Categorical(logit).sample().item()
        return action, logit, hidden_out

    def reset_hidden_states(self):
        return self.actor.reset_hidden_states()

    @property
    def optim(self):
        return self._actor_optim


class COMAAgentGroup:

    def __init__(
        self,
        input_size,
        output_size,
        actor_leraning_rate=3e-4,
        critic_learning_rate=1e-3,
        n=3,
        gamma=0.99
    ):
        self.actors = [COMAAgent(input_size, output_size, learning_rate=actor_leraning_rate) for _ in range(n)]
        self.critic = Critic(input_size * n, output_size, n=n)
        # self.critic_target = Critic(input_size, output_size)
        self._critic_optim = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self._output_size = output_size

        self._n = n
        self._gamma = gamma

    def get_action(self, observations, hidden_states):
        observations = torch.tensor(observations).float()
        return tuple(actor.get_action(observation, h_in)
                     for actor, observation, h_in in zip(self.actors, observations, hidden_states))

    def value(self, state):
        return self.critic(state)

    def _cross_prod(self, *args):
        # FIXME: x.T @ y
        new_pi = torch.zeros(1, self._output_size[0] ** len(args))
        for i in range(self._output_size[0]):
            for j in range(self._output_size[0]):
                for k in range(self._output_size[0]):
                    new_pi[0, i*self._output_size[0]**2 + j*self._output_size[0] + k] = args[0][i] * args[1][j] * args[2][k]
        return new_pi

    def reset_hidden_states(self):
        return (actor.reset_hidden_states() for actor in self.actors)

    def train(self, states, actions, logits, rewards):
        group_state = torch.cat(tuple(map(lambda x: torch.tensor(x).float(), states)))
        q_value = self.value(group_state)
        q_est = q_value.clone()
        for t in range(len(rewards)-1):
            a_index = sum([actions[i][t] * (self.n ** (2-i)) for i in range(len(actions))])
            q_est[t][a_index] = rewards[t] + self.gamma * torch.sum(self._cross_prod(actions[0][t+1], actions[1][t+1], actions[2][t+1]) * q_est[t+1, :])
        a_index = actions[0][-1] * self._output_size[0] ** 2 + actions[1][-1] * self._output_size[0] + actions[2][-1]
        q_est[-1][a_index] = rewards[-1]
        value_loss = F.smooth_l1_loss(q_value, q_est.detach())
        self._critic_optim.zero_grad()
        value_loss.backward()
        self._critic_optim.step()

        for t in range(len(rewards)):
            temp_q = torch.zeros(1, self.n)
            for a in range(self.n):
                pass

    @property
    def optim(self):
        return self._critic_optim

    @property
    def n(self):
        return self._n

    @property
    def gamma(self):
        return self._gamma


def main():
    input_size = 70
    output_size = 5
    n = 3
    time_horizon = 128
    batch_size = 32

    inputs = np.random.uniform(-1.0, 1.0, size=(input_size,))
    inputs = torch.tensor(inputs).float()

    group = COMAAgentGroup(input_size, output_size, n=n)
    """
    hidden_states = group.reset_hidden_states()
    value = group.value(inputs)
    inputs = torch.cat([inputs.unsqueeze(0)] * n, dim=0)
    actions = group.get_action(inputs, hidden_states)
    for action in actions:
        print(f'action: {action[0]}, logit: {action[1]}, h_out: {action[2].shape}')
    print('value:', value)
    return
    """

    """
    critic = Critic(input_size, output_size)
    value = critic(inputs)
    print('value:', value)

    agent = COMAAgent(input_size, output_size)
    hidden_in = agent.hidden_state()
    action, logit, hidden_out = agent.get_action(inputs, hidden_in)

    print('inputs:', inputs.shape)
    print('hidden_in:', hidden_in.shape)
    print('hidden_out:', hidden_out.shape)
    # print('outputs:', outputs.shape)
    print('logits:', logit)
    print('action:', action)
    print(np.all(hidden_in.detach().numpy() == hidden_out.detach().numpy()))
    """

    # Pseudo Code
    for episode in count(1):
        print(f'Episode: {episode}')
        observations = [[] for _ in range(n)]
        hidden_states = group.reset_hidden_states()

        for t in range(time_horizon):
            observations = torch.cat([inputs.unsqueeze(0)] * n, dim=0)
            actions = group.get_action(observations, hidden_states)

            if done := True:
                break

        if episode > 0:
            break


if __name__ == "__main__":
    main()
