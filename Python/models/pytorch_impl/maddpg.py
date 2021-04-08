#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.pi = nn.Linear(hidden_size, output_size)

        self.max_action = 1
        # self.min_action = -1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # policy = F.softmax(x)
        x = self.max_action * torch.tanh(self.pi(x))
        return x


class Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

        self.max_action = 1
        # self.min_action = -1

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.value(x)
        return q_value


class MADDPGAgent:

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=64,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        tau=0.01,
        gamma=0.95,
        id_=0
    ):
        self.actor = Actor(state_size, action_size, hidden_size=hidden_size)
        self.critic = Critic(state_size, action_size, hidden_size=hidden_size)

        self.actor_target = Actor(state_size, action_size, hidden_size=hidden_size)
        self.critic_target = Critic(state_size, action_size, hidden_size=hidden_size)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self._tau = tau
        self._gamma = gamma
        self._id = id_

    def get_action(self, state, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.uniform()

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def train(self, transitions, other_agents):
        # FIXME
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions[f'r_{self.id}']
        s, a, s_next = [], [], []
        for agent_id in range(self.n_agents):
            s.append(transitions[f's_{agent_id}'])
            a.append(transitions[f'a_{agent_id}'])
            s_next.append(transitions[f's_next_{agent_id}'])

        a_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(self.n_agents):
                if agent_id == self.id:
                    a_next.append(self.actor_target(s_next[agent_id]))
                else:
                    a_next.append(other_agents[index].policy.actor_target(s_next[agent_id]))
                    index += 1
            q_next = self.critic_target(s_next, a_next).detach()
            target_q = (r.unsqueeze(1) + self.gamma * q_next).detach()

        q_value = self.critic(s, a)
        critic_loss = (target_q - q_value).pow(2).mean()

        a[self.id] = self.actor(o[self.id])
        actor_loss = -self.critic(s, a).mean()

        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()

        return actor_loss_value, critic_loss_value

    @property
    def tau(self):
        return self._tau

    @property
    def gamma(self):
        return self._gamma

    @property
    def id(self):
        return self._id


def main():
    pass


if __name__ == "__main__":
    main()
