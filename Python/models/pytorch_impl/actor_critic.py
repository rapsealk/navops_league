#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing
from torch.distributions import Categorical

from .mask import BooleanMaskLayer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def weights_init_(m, activation_fn=F.relu):
    if isinstance(m, nn.Linear):
        # ReLU
        if activation_fn is F.relu:
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)
        """
        # SiLU
        elif activation_fn is F.silu:
            torch.nn.init.xavier_normal_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
        """


def convert_to_tensor(device, *args):
    return map(lambda tensor: tensor.float().to(device), map(torch.tensor, args))


class MultiHeadLSTMActorCriticModel(nn.Module):

    def __init__(self, input_size, output_sizes, hidden_size=512, rnn_hidden_size=64, rnn_num_layers=1):
        super(MultiHeadLSTMActorCriticModel, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_num_layers = rnn_num_layers

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear4 = nn.Linear(hidden_size * 2, hidden_size)
        self.rnn = nn.LSTM(hidden_size, rnn_hidden_size, num_layers=rnn_num_layers, batch_first=True)

        maneuver_action_size, attack_action_size = output_sizes

        self.maneuver_h = nn.Linear(rnn_hidden_size, hidden_size)
        self.maneuver_h2 = nn.Linear(hidden_size, hidden_size)
        self.actor_maneuver = nn.Linear(hidden_size, maneuver_action_size)
        self.critic_maneuver = nn.Linear(hidden_size, 1)

        self.attack_h = nn.Linear(rnn_hidden_size, hidden_size)
        self.attack_h2 = nn.Linear(hidden_size, hidden_size)
        self.actor_attack = nn.Linear(hidden_size, attack_action_size)
        self.critic_attack = nn.Linear(hidden_size, 1)

        # self.mask = BooleanMaskLayer(output_sizes[0])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.apply(weights_init_)

    def forward(self, x, h_in):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        x_h, h_out = self.rnn(x, h_in)

        x_m = F.relu(self.maneuver_h(x_h))
        x_m = F.relu(self.maneuver_h2(x_m))
        logits_m = self.actor_maneuver(x_m)
        value_m = self.critic_maneuver(x_m)

        x_a = F.relu(self.attack_h(x_h))
        x_a = F.relu(self.attack_h2(x_a))
        logits_a = self.actor_attack(x_a)
        value_a = self.critic_attack(x_a)

        return (logits_m, logits_a), (value_m, value_a), h_out

    def reset_hidden_state(self, batch_size=1):
        hidden_states = (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size),
                         torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size))
        return tuple(map(lambda x: x.to(self.device), hidden_states))

    @property
    def rnn_num_layers(self):
        return self._rnn_num_layers

    @property
    def rnn_hidden_size(self):
        return self._rnn_hidden_size

    @property
    def device(self):
        return self._device

    def to(self, device):
        # self.mask = self.mask.to(device)
        self._device = device
        return super(MultiHeadLSTMActorCriticModel, self).to(device)


class SharedAdam(optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class MultiHeadLSTMActorCriticAgent:

    def __init__(
        self,
        input_size,
        output_sizes,
        hidden_size=512,
        rnn_hidden_size=64,
        rnn_num_layers=1,
        learning_rate=3e-5,
        cuda=True
    ):
        if cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        self._model = MultiHeadLSTMActorCriticModel(input_size,
                                                    output_sizes,
                                                    hidden_size=hidden_size,
                                                    rnn_hidden_size=rnn_hidden_size,
                                                    rnn_num_layers=rnn_num_layers)
        self._target_model = MultiHeadLSTMActorCriticModel(input_size,
                                                    output_sizes,
                                                    hidden_size=hidden_size,
                                                    rnn_hidden_size=rnn_hidden_size,
                                                    rnn_num_layers=rnn_num_layers)
        self._model = self.model.to(device)
        self._target_model = self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # self._optim = optim.SAdam(self._model.parameters(), lr=learning_rate)
        self._optim = SharedAdam(self._model.parameters(), lr=learning_rate)

    def get_action(self, state, hidden):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
            state = state.view(-1, 1, state.shape[-1]).to(self.device)
        hidden = tuple(h_in.to(self.device) for h_in in hidden)

        logits, _, h_out = self._model(state, hidden)
        probs_m = F.softmax(logits[0], dim=-1)
        action_maneuver = Categorical(probs_m).sample().item()
        probs_a = F.softmax(logits[1], dim=-1)
        action_attack = Categorical(probs_a).sample().item()
        h_out = tuple(h_o.detach().cpu() for h_o in h_out)

        return ((action_maneuver, probs_m),
                (action_attack, probs_a),
                h_out)

    def loss(self, states, actions, rewards):
        self._model.train()

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)

        h_ins = self.model.reset_hidden_state(batch_size=len(states))

        logits, values, h_outs = self._model(states, h_ins)
        # td = values - rewards
        # critic_loss = td.pow(2)
        td_m = values[0] - rewards
        td_a = values[1] - rewards
        critic_loss = td_m.pow(2) + td_a.pow(2)

        # Target Model
        _, values, _ = self.target_model(states, h_ins)
        td_m = values[0] - rewards
        td_a = values[1] - rewards

        prob_m = F.softmax(logits[0], dim=-1)
        action_m = Categorical(prob_m)
        # action_m = prob_m.gather(-1, actions[:, 0])
        exp_v = action_m.log_prob(actions[:, 0]) * td_m.detach().squeeze()
        actor_loss_m = -exp_v

        prob_a = F.softmax(logits[1], dim=-1)
        action_a = Categorical(prob_a)
        exp_v = action_a.log_prob(actions[:, 1]) * td_a.detach().squeeze()
        actor_loss_a = -exp_v

        total_loss = 0.5 * critic_loss.mean() + actor_loss_m.mean() + actor_loss_a.mean()

        return total_loss

    def apply_gradients(self, grad_agent, loss):
        print(f'[A3C] Source device: {grad_agent.device}, Target device: {self.device} ...')
        self.optim.zero_grad()
        grad_agent.optim.zero_grad()
        loss.backward()

        for param, target_param in zip(grad_agent.model.parameters(), self.model.parameters()):
            target_param._grad = param.grad.to(self.device)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)

        self.optim.step()

    def reset_hidden_state(self, batch_size=1):
        return self.model.reset_hidden_state(batch_size=batch_size)

    def hard_update(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, tau=1e-2):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1 - tau))

    def state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def save(self, path: str, episode: int = 0):
        pathlib.Path(os.path.abspath(os.path.dirname(path))).mkdir(parents=True, exist_ok=True)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        torch.save({
            "params": self._model.state_dict(),
            # "optim": self._optim.parameters(),
            "episode": episode
            # TODO: epsilon
        }, path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self._model.load_state_dict(state_dict["params"])
        # self._optim.load(state_dict["optim"])
        return state_dict.get("episode", 0)

    def to(self, device):
        self._model = self.model.to(device)
        self._target_model = self.target_model.to(device)
        return self

    def share_memory(self):
        self.model.share_memory()
        # self.optim.share_memory()

    @property
    def model(self):
        return self._model

    @property
    def target_model(self):
        return self._target_model

    @property
    def optim(self):
        return self._optim

    @property
    def device(self):
        return self.model.device


if __name__ == "__main__":
    pass
