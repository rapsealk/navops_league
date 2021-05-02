#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.distributions import Categorical

from algorithms import MADDPG


class Agent:
    def __init__(self, input_size, output_size, agent_id, n=3):
        self._output_size = output_size
        self.agent_id = agent_id
        self.policy = MADDPG(input_size, output_size, agent_id=agent_id, n=n)

    def select_action(self, o, h_in, noise_rate=0.0, epsilon=0.01):
        #if np.random.uniform() < epsilon:
        #    # u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        #    u = np.random.randint(0, self.output_size)
        #else:
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        prob, h_out = self.policy.actor_network(inputs, h_in)   # .squeeze(0)
        action = Categorical(prob).sample()
        # print('{} : {}'.format(self.name, pi))
        action = action.cpu().detach().item()   # .numpy()
        # noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
        # u += noise
        # u = np.clip(u, -self.args.high_action, self.args.high_action)
        # return u.copy()
        return action, h_out

    def learn(self, transitions, other_agents):
        return self.policy.train(transitions, other_agents)

    def reset_hidden_states(self, batch_size=8):
        return self.policy.reset_hidden_states(batch_size=batch_size)

    def save(self, path):
        self.policy.save(path)

    @property
    def output_size(self):
        return self._output_size


def main():
    pass


if __name__ == "__main__":
    main()
