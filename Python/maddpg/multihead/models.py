#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import onehot, weights_init_

STATE_STEER_LEFT_MAX = -5
STATE_STEER_RIGHT_MAX = -1
STATE_ENGINE_BACKWARD_MAX = -10
STATE_ENGINE_FORWARD_MAX = -6

ACTION_ENGINE_FORWARD = 1
ACTION_ENGINE_BACKWARD = 2
ACTION_STEER_LEFT = 3
ACTION_STEER_RIGHT = 4


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


class MultiHeadActorCritic(nn.Module):

    def __init__(self, input_size, output_sizes, hidden_size=1024, rnn_hidden_size=512, batch_size=8):
        super(MultiHeadActorCritic, self).__init__()
        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size
        self._batch_size = batch_size

        movement_action_size, attack_action_size = output_sizes
        self._movement_action_size = movement_action_size
        self._attack_action_size = attack_action_size

        self.rnn = nn.GRUCell(input_size, rnn_hidden_size)

        self.actor_movement = nn.Sequential(
            nn.Linear(rnn_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, movement_action_size)
        )

        self.critic_movement = nn.Sequential(
            nn.Linear(rnn_hidden_size+movement_action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor_attack = nn.Sequential(
            nn.Linear(rnn_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, attack_action_size)
        )

        self.critic_attack = nn.Sequential(
            nn.Linear(rnn_hidden_size+attack_action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.movement_action_mask = BooleanMaskLayer(movement_action_size)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.apply(weights_init_)

    def forward(self, x, h_in):
        h_out = self.rnn(x, h_in)

        p_movement = self.actor_movement(h_out) * self.movement_action_mask(x)

        p_movement = F.softmax(p_movement, dim=-1)
        oh_movement = onehot(torch.argmax(p_movement, dim=-1), max_range=self.movement_action_size)

        v_movement = self.critic_movement(torch.cat([h_out, oh_movement.float()], dim=-1))

        p_attack = self.actor_attack(h_out)
        p_attack = F.softmax(p_attack, dim=-1)
        oh_attack = onehot(torch.argmax(p_attack, dim=-1), max_range=self.attack_action_size)

        v_attack = self.critic_attack(torch.cat([h_out, oh_attack.float()], dim=-1))

        return (p_movement, v_movement), (p_attack, v_attack), h_out

    def to(self, device):
        self._device = device
        self.movement_action_mask = self.movement_action_mask.to(device)
        return super(MultiHeadActorCritic, self).to(device)

    def reset_hidden_state(self, batch_size=1):
        return torch.zeros((batch_size, self.rnn_hidden_size))

    @property
    def device(self):
        return self._device

    """
    @property
    def hidden_size(self):
        return self._hidden_size
    """

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size

    @property
    def movement_action_size(self):
        return self._movement_action_size

    @property
    def attack_action_size(self):
        return self._attack_action_size


def main():
    rollout = 4
    action_sizes = (8, 2)
    inputs = np.random.uniform(-1.0, 1.0, (rollout, 24))

    model = MultiHeadActorCritic(inputs.shape[-1], action_sizes)
    h_in = model.reset_hidden_state(batch_size=rollout)

    inputs = torch.from_numpy(inputs).float()
    print('inputs:', inputs.shape)
    (p_m, v_m), (p_a, v_a), h_out = model(inputs, h_in)
    print(f'p_m: {p_m.shape}, v_m: {v_m.shape}, p_a: {p_a.shape}, v_a: {v_a.shape}')
    print('h_in:', h_in.shape, 'h_out:', h_out.shape)

    action_m = Categorical(probs=p_m[-1]).sample().numpy()
    action_a = Categorical(probs=p_a[-1]).sample().numpy()
    print('action:', action_m, action_a)


if __name__ == "__main__":
    main()
