#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

STATE_STEER_LEFT_MAX = -5
STATE_STEER_RIGHT_MAX = -1
STATE_ENGINE_BACKWARD_MAX = -10
STATE_ENGINE_FORWARD_MAX = -6

ACTION_ENGINE_FORWARD = 1
ACTION_ENGINE_BACKWARD = 2
ACTION_STEER_LEFT = 3
ACTION_STEER_RIGHT = 4


def onehot(items, max_range=10):
    # print('onehot.items.shape:', items.shape, 'max_range:', max_range)
    x = np.zeros((items.shape[0], max_range))
    x[range(items.shape[0]), items] = 1
    return torch.from_numpy(x).view(-1)


class BooleanMaskLayer(nn.Module):

    def __init__(self, output_size):
        super(BooleanMaskLayer, self).__init__()
        self._output_size = output_size

    def forward(self, x: torch.Tensor):
        masking = -1e9  # float("-inf")
        x = x.clone().detach().cpu().squeeze().numpy()
        # Steer: -3 ~ -7 (-3, +4)
        # Speed: -8 ~ -12 (+1, -2)
        if x.ndim == 1:
            mask = np.ones(self._output_size)
            if x[STATE_ENGINE_BACKWARD_MAX] == 1.0:
                mask[ACTION_ENGINE_BACKWARD] = masking
            elif x[STATE_ENGINE_FORWARD_MAX] == 1.0:
                mask[ACTION_ENGINE_FORWARD] = masking
            elif x[STATE_STEER_LEFT_MAX] == 1.0:
                mask[ACTION_STEER_LEFT] = masking
            elif x[STATE_STEER_RIGHT_MAX] == 1.0:
                mask[ACTION_STEER_RIGHT] = masking
            mask = torch.tensor(mask, requires_grad=False)
        elif x.ndim == 2:
            mask = np.ones((x.shape[0], self._output_size))
            mask[np.where(x[:, STATE_ENGINE_BACKWARD_MAX] == 1.0), ACTION_ENGINE_BACKWARD] = masking
            mask[np.where(x[:, STATE_ENGINE_FORWARD_MAX] == 1.0), ACTION_ENGINE_FORWARD] = masking
            mask[np.where(x[:, STATE_STEER_LEFT_MAX] == 1.0), ACTION_STEER_LEFT] = masking
            mask[np.where(x[:, STATE_STEER_RIGHT_MAX] == 1.0), ACTION_STEER_RIGHT] = masking
            mask = torch.tensor(mask, requires_grad=False)  # .unsqueeze(1)

        return mask


class Actor(nn.Module):

    def __init__(self, input_size, action_sizes, hidden_size=64, rnn_hidden_size=64, batch_size=8):
        super(Actor, self).__init__()
        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size
        self._batch_size = batch_size

        movement_action_size, attack_action_size = action_sizes

        self.rnn = nn.GRUCell(input_size, rnn_hidden_size)
        self.linear1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.2)

        self.movement_linear1 = nn.Linear(hidden_size, hidden_size)
        self.movement_dropout1 = nn.Dropout(p=0.2)
        self.movement_linear2 = nn.Linear(hidden_size, hidden_size)
        self.movement_dropout2 = nn.Dropout(p=0.2)
        self.movement_linear3 = nn.Linear(hidden_size, hidden_size)
        self.movement_linear4 = nn.Linear(hidden_size, movement_action_size)

        self.attack_linear1 = nn.Linear(hidden_size, hidden_size)
        self.attack_dropout1 = nn.Dropout(p=0.2)
        self.attack_linear2 = nn.Linear(hidden_size, hidden_size)
        self.attack_dropout2 = nn.Dropout(p=0.2)
        self.attack_linear3 = nn.Linear(hidden_size, hidden_size)
        self.attack_linear4 = nn.Linear(hidden_size, attack_action_size)

        self.action_mask = BooleanMaskLayer(movement_action_size)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, h_in):
        # print(f'[ACTOR] forward(x={x.shape}, h_in={h_in.shape})')
        x = x.to(self.device)
        h_in = h_in.to(self.device)
        h_out = self.rnn(x, h_in)
        y = F.relu(self.linear1(h_out))
        y = self.dropout1(y)
        y = F.relu(self.linear2(y))
        y = self.dropout2(y)

        y_m = F.relu(self.movement_linear1(y))
        y_m = self.movement_dropout1(y_m)
        y_m = F.relu(self.movement_linear2(y_m))
        y_m = self.movement_dropout2(y_m)
        y_m = F.relu(self.movement_linear3(y_m))
        y_m = self.movement_linear4(y_m) * self.action_mask(x).to(self.device)
        prob_m = F.softmax(y_m, dim=-1)

        y_a = F.relu(self.attack_linear1(y))
        y_a = self.attack_dropout1(y_a)
        y_a = F.relu(self.attack_linear2(y_a))
        y_a = self.attack_dropout2(y_a)
        y_a = F.relu(self.attack_linear3(y_a))
        y_a = self.attack_linear4(y_a)
        prob_a = F.softmax(y_a, dim=-1)

        return prob_m, prob_a, h_out

    def to(self, device):
        self._device = device
        return super(Actor, self).to(device)

    def reset_hidden_state(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return torch.zeros((batch_size, self.rnn_hidden_size))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device


class Critic(nn.Module):

    def __init__(self, input_size, action_sizes, hidden_size=64, rnn_hidden_size=64, batch_size=8):
        super(Critic, self).__init__()
        self._action_sizes = action_sizes
        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size
        self._batch_size = batch_size

        movement_action_size, attack_action_size = action_sizes

        self.rnn = nn.GRUCell(input_size, rnn_hidden_size)
        self.linear1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.2)

        self.movement_linear1 = nn.Linear(hidden_size, hidden_size)
        self.movement_dropout1 = nn.Dropout(p=0.2)
        self.movement_linear2 = nn.Linear(hidden_size, hidden_size)
        self.movement_dropout2 = nn.Dropout(p=0.2)
        self.movement_linear3 = nn.Linear(hidden_size, hidden_size)
        self.q_movement = nn.Linear(hidden_size, 1)

        self.attack_linear1 = nn.Linear(hidden_size, hidden_size)
        self.attack_dropout1 = nn.Dropout(p=0.2)
        self.attack_linear2 = nn.Linear(hidden_size, hidden_size)
        self.attack_dropout2 = nn.Dropout(p=0.2)
        self.attack_linear3 = nn.Linear(hidden_size, hidden_size)
        self.q_attack = nn.Linear(hidden_size, 1)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state, actions, h_in):
        # state = torch.cat(state, dim=1)
        state = state.view(state.shape[0], -1).to(self.device)
        # for i in range(len(action)):
        #     action[i] /= self.max_action
        # action = torch.cat(action, dim=1)

        move_action = torch.transpose(actions[:, 0].squeeze().byte(), dim0=-1, dim1=0)
        attack_action = torch.transpose(actions[:, 1].squeeze().byte(), dim0=-1, dim1=0)
        oh_action = []
        for mov_a, atk_a in zip(move_action, attack_action):
            oh_mov_a = onehot(mov_a.cpu().numpy(), max_range=self.action_sizes[0])
            oh_atk_a = onehot(atk_a.cpu().numpy(), max_range=self.action_sizes[1])
            oh_action.append(torch.cat([oh_mov_a, oh_atk_a]))
        action = torch.stack(oh_action).to(self.device)

        x = torch.cat([state, action], dim=-1).float().to(self.device)
        h_in = h_in.to(self.device)
        h_out = self.rnn(x, h_in)
        x = F.relu(self.linear1(h_out))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)

        x_m = F.relu(self.movement_linear1(x))
        x_m = self.movement_dropout1(x_m)
        x_m = F.relu(self.movement_linear2(x_m))
        x_m = self.movement_dropout2(x_m)
        x_m = F.relu(self.movement_linear3(x_m))
        q_movement = torch.tanh(self.q_movement(x_m))

        x_a = F.relu(self.attack_linear1(x))
        x_a = self.attack_dropout1(x_a)
        x_a = F.relu(self.attack_linear2(x_a))
        x_a = self.attack_dropout2(x_a)
        x_a = F.relu(self.attack_linear3(x_a))
        q_attack = torch.tanh(self.q_attack(x_a))

        return q_movement, q_attack, h_out

    def to(self, device):
        self._device = device
        return super(Critic, self).to(device)

    def reset_hidden_state(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return torch.zeros((batch_size, self.rnn_hidden_size))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def action_sizes(self):
        return self._action_sizes

    @property
    def device(self):
        return self._device


def main():
    rollout = 4
    action_size = 10
    inputs = np.random.uniform(-1.0, 1.0, (rollout, 24))
    pi = Actor(input_size=inputs.shape[-1], output_size=action_size, hidden_size=32, rnn_hidden_size=64, batch_size=rollout)
    h_in = pi.reset_hidden_state()

    inputs = torch.from_numpy(inputs).float()
    print('inputs:', inputs.shape)
    logits, h_out = pi(inputs, h_in)
    print('logits:', logits.shape)
    print('h_in:', h_in.shape, 'h_out:', h_out.shape)

    action = Categorical(logits=logits[-1]).sample().numpy()
    print('action:', action)

    print('---')

    critic = Critic(inputs.shape[-1] * rollout + action_size, action_size, hidden_size=32, rnn_hidden_size=64, batch_size=1)
    critic = critic.to(critic.device)
    h_in = critic.reset_hidden_state(batch_size=1)

    inputs = inputs.unsqueeze(0)
    actions = torch.from_numpy(action).unsqueeze(0).unsqueeze(0)
    q, h_out = critic(inputs, actions, h_in)
    print('q:', q)
    print('h_in:', h_in.shape, 'h_out:', h_out.shape)


if __name__ == "__main__":
    main()
