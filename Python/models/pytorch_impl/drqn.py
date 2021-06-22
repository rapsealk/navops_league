#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.distributions import Categorical
import numpy as np

# from .mask import BooleanMaskLayer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def weights_init_(m, activation_fn=F.relu):
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
    # return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))
    return map(lambda tensor: tensor.float().to(device), map(torch.tensor, args))


class DRQN(nn.Module):

    def __init__(self, input_size, action_sizes=(5, 2), hidden_size=512, rnn_hidden_size=64):
        """
        action_sizes: tuple(int, int)
        - maneuver action size and attack action size.
        """
        super(DRQN, self).__init__()

        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size
        self._action_sizes = action_sizes

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, self.rnn_hidden_size, batch_first=True)
        self.maneuver_h = nn.Linear(self.rnn_hidden_size, action_sizes[0])
        self.attack_h = nn.Linear(self.rnn_hidden_size+action_sizes[0], action_sizes[1])

        self.apply(weights_init_)

    def forward(self, x, h_in):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x, h_out = self.lstm(x, h_in)
        logits_m = self.maneuver_h(x)
        logits_a = self.attack_h(torch.cat([x, logits_m], dim=-1))
        return logits_m, logits_a, h_out

    def reset_hidden_state(self, batch_size=32):
        hidden_states = (torch.zeros(1, batch_size, self.rnn_hidden_size),
                         torch.zeros(1, batch_size, self.rnn_hidden_size))
        return tuple(map(lambda x: x.to(self.device), hidden_states))

    def to(self, device):
        self._device = device
        return super(DRQN, self).to(device)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size

    @property
    def action_sizes(self):
        return self._action_sizes

    @property
    def device(self):
        return self._device


class DRQNAgent:

    def __init__(self,
                 input_size,
                 action_sizes=(5, 2),
                 hidden_size=512,
                 rnn_hidden_size=64,
                 batch_size=8,
                 gamma=0.98,
                 learning_rate=1e-3):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = DRQN(input_size=input_size,
                           action_sizes=action_sizes,
                           hidden_size=hidden_size,
                           rnn_hidden_size=rnn_hidden_size).to(self._device)
        self._target_model = DRQN(input_size=input_size,
                                  action_sizes=action_sizes,
                                  hidden_size=hidden_size,
                                  rnn_hidden_size=rnn_hidden_size).to(self._device)
        self._target_model.load_state_dict(self._model.state_dict())
        self._target_model.eval()

        self.optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        self._batch_size = 8
        self._gamma = gamma

    def get_actions(self, x, h_in, epsilon=0.0):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.view(-1, 1, x.shape[-1]).to(self.device)
        logits_m, logits_a, h_out = self._model(x, h_in)
        if np.random.random() > epsilon:
            action_m = F.softmax(logits_m, dim=-1)
            action_m = torch.argmax(action_m, dim=-1).squeeze().item()
            action_a = F.softmax(logits_a, dim=-1)
            action_a = torch.argmax(action_a, dim=-1).squeeze().item()
            actions = (action_m, action_a)
        else:
            actions = (np.random.randint(self.action_sizes[0]),
                       np.random.randint(self.action_sizes[1]))
        return actions, (logits_m.detach(), logits_a.detach()), h_out

    def train(self,
              episode_memory=None,
              learning_rate=1e-3,
              gamma=None,
              batch_size=None):

        gamma = gamma or self.gamma
        batch_size = batch_size or self.batch_size

        # Get batch from replay buffer
        samples, sequence_length = episode_memory.sample()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        # for i in range(len(samples)):
        for sample in samples:
            obs, action, r, next_obs, done = sample
            observations.extend(obs)
            actions.extend(action)
            rewards.extend(r)
            next_observations.extend(next_obs)
            dones.extend(done)

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        # seq_len = observations.shape[0] // args.batch_size

        observations = torch.FloatTensor(observations.reshape(batch_size, sequence_length, -1)).to(self.device)
        actions_m = torch.LongTensor(actions[:, 0].reshape(batch_size, sequence_length, -1)).to(self.device)
        actions_a = torch.LongTensor(actions[:, 1].reshape(batch_size, sequence_length, -1)).to(self.device)
        rewards = torch.FloatTensor(rewards.reshape(batch_size, sequence_length, -1)).to(self.device)
        next_observations = torch.FloatTensor(next_observations.reshape(batch_size, sequence_length, -1)).to(self.device)
        dones = torch.FloatTensor(dones.reshape(batch_size, sequence_length, -1)).to(self.device)

        h_in = self._target_model.reset_hidden_state(batch_size=batch_size)
        h_in = tuple([h.to(self.device) for h in h_in])

        q_target_m, q_target_a, _ = self._target_model(next_observations, h_in)

        h_in = self._model.reset_hidden_state(batch_size=batch_size)
        q_out_m, q_out_a, _ = self._model(observations, h_in)

        # 1. Maneuver(Q)
        q_target_m_max = q_target_m.max(2)[0].view(batch_size, sequence_length, -1).detach()
        targets_m = rewards + gamma * q_target_m_max * dones
        q_action_m = q_out_m.gather(dim=2, index=actions_m)
        loss = F.smooth_l1_loss(q_action_m, targets_m)

        # 2. Attacl(Q)
        q_target_a_max = q_target_a.max(2)[0].view(batch_size, sequence_length, -1).detach()
        targets_a = rewards + gamma * q_target_a_max * dones
        q_action_a = q_out_a.gather(dim=2, index=actions_a)
        loss += F.smooth_l1_loss(q_action_a, targets_a)

        # Update Network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def hard_update(self):
        for param, target_param in zip(self._model.parameters(), self._target_model.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, tau=1e-2):
        for param, target_param in zip(self._model.parameters(), self._target_model.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1 - tau))

    def reset_hidden_state(self, batch_size=32):
        return self._model.reset_hidden_state(batch_size=batch_size)

    def save(self, path: str, episode: int = 0, epsilon=0.01):
        pathlib.Path(os.path.abspath(os.path.dirname(path))).mkdir(parents=True, exist_ok=True)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        torch.save({
            "params": self._model.state_dict(),
            # "optim": self._optim.parameters(),
            "episode": episode,
            "epsilon": epsilon
        }, path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self._model.load_state_dict(state_dict["params"])
        self._target_model.load_state_dict(state_dict["params"])
        # self._optim.load(state_dict["optim"])
        return {
            "episode": state_dict.get("episode", 0),
            "epsilon": state_dict.get("epsilon", 0.01)
        }

    @property
    def action_sizes(self):
        return self._model.action_sizes

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def gamma(self):
        return self.gamma

    @property
    def device(self):
        return self._model.device


if __name__ == "__main__":
    pass
