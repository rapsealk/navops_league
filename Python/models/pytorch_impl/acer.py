#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .mask import BooleanMaskLayer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
    # return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))
    return map(lambda tensor: tensor.float().to(device), map(torch.tensor, args))


class MultiHeadLstmActorCriticModel(nn.Module):

    def __init__(self, input_size, output_sizes, hidden_size=256):
        super(MultiHeadLstmActorCriticModel, self).__init()
        self._rnn_input_size = 256
        self._rnn_output_size = 128

        self.encoder = nn.Linear(input_size, self._rnn_input_size)
        self.rnn = nn.LSTM(self._rnn_input_size, self._rnn_output_size)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self._rnn_output_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        movement_action_size, attack_action_size = output_sizes

        self.actor_movement_h = nn.Linear(hidden_size, hidden_size)
        self.actor_movement_h2 = nn.Linear(hidden_size, hidden_size)
        self.actor_movement = nn.Linear(hidden_size, movement_action_size)

        self.critic_movement_h = nn.Linear(hidden_size, hidden_size)
        self.critic_movement_h2 = nn.Linear(hidden_size, hidden_size)
        self.critic_movement = nn.Linear(hidden_size, movement_action_size)

        self.actor_attack_h = nn.Linear(hidden_size, hidden_size)
        self.actor_attack_h2 = nn.Linear(hidden_size, hidden_size)
        self.actor_attack = nn.Linear(hidden_size, attack_action_size)

        self.critic_attack_h = nn.Linear(hidden_size, hidden_size)
        self.critic_attack_h2 = nn.Linear(hidden_size, hidden_size)
        self.critic_attack = nn.Linear(hidden_size, attack_action_size)

        # self.mask = BooleanMaskLayer(output_size)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.apply(weights_init_)

    def forward(self, x, hidden):
        x = F.silu(self.encoder(x))
        x = x.view(-1, 1, self._rnn_input_size)
        x, hidden = self.rnn(x, hidden)
        x = F.silu(self.linear(x))
        x = F.silu(self.linear2(x))

        x_p_move = F.silu(self.actor_movement_h(x))
        x_p_move = F.silu(self.actor_movement_h2(x_p_move))
        logit_movement = self.actor_movement(x_p_move)

        x_v_move = F.silu(self.critic_movement_h(x))
        x_v_move = F.silu(self.critic_movement_h2(x_v_move))
        value_movement = self.critic_movement(x_v_move)

        x_p_attack = F.silu(self.actor_attack_h(x))
        x_p_attack = F.silu(self.actor_attack_h2(x_p_attack))
        logit_attack = self.actor_attack(x_p_attack)

        x_v_attack = F.silu(self.critic_attack_h(x))
        x_v_attack = F.silu(self.critic_attack_h2(x_v_attack))
        value_attack = self.critic_movement(x_v_attack)

        return (logit_movement, value_movement), (logit_attack, value_attack), hidden

    def get_policy(self, x, hidden):
        x = F.silu(self.encoder(x))
        x = x.view(-1, 1, self._rnn_input_size)
        x, hidden = self.rnn(x, hidden)
        x = F.silu(self.linear(x))
        x = F.silu(self.linear2(x))

        x_p_move = F.silu(self.actor_movement_h(x))
        x_p_move = F.silu(self.actor_movement_h2(x_p_move))
        logit_movement = self.actor_movement(x_p_move)
        prob_movement = F.softmax(logit_movement, dim=2)

        x_p_attack = F.silu(self.actor_attack_h(x))
        x_p_attack = F.silu(self.actor_attack_h2(x_p_attack))
        logit_attack = self.actor_attack(x_p_attack)
        prob_attack = F.softmax(logit_attack, dim=2)

        return prob_movement, prob_attack, hidden

    def value(self, x, hidden):
        x = F.silu(self.encoder(x))
        x = x.view(-1, 1, self._rnn_input_size)
        x, _ = self.rnn(x, hidden)
        x = F.silu(self.linear(x))
        x = F.silu(self.linear2(x))

        x_v_move = F.silu(self.critic_movement_h(x))
        x_v_move = F.silu(self.critic_movement_h2(x_v_move))
        value_movement = self.critic_movement(x_v_move)

        x_v_attack = F.silu(self.critic_attack_h(x))
        x_v_attack = F.silu(self.critic_attack_h2(x_v_attack))
        value_attack = self.critic_movement(x_v_attack)

        return value_movement, value_attack


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
        self.critic = nn.Linear(hidden_size, output_size)

        self.mask = BooleanMaskLayer(output_size)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.apply(weights_init_)

    def forward(self, x, hidden):
        # silu, relu
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

    def get_policy(self, inputs, hidden, training=False):
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
        if not training:
            x = x + self.mask(inputs).to(self._device)
        #mask = self.mask(inputs).to(self._device)
        #x = x + mask
        # x = x + self.mask(inputs).to(self._device)
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


class MultiHeadAcerAgent:

    def __init__(
        self,
        model,
        buffer,
        c_sampling_ratio=1.0,
        learning_rate=3e-5,
        cuda=True
    ):
        if cuda:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device("cpu")

        self._model = model.to(self._device)
        self._optim = optim.Adam(self._model.parameters(), lr=learning_rate)

        self._gamma = 0.98
        self._c_sampling_ratio = c_sampling_ratio

        self._buffer = buffer

    def get_action(self, state, hidden):
        state = torch.from_numpy(state).float().to(self._device)
        hidden = tuple(h_in.to(self._device) for h_in in hidden)
        prob_movement, prob_attack, hidden_out = self._model.get_policy(state, hidden)
        del state, hidden
        action_movement = Categorical(prob_movement).sample().item()
        prob_movement = prob_movement.detach().cpu().numpy().squeeze()
        action_attack = Categorical(prob_attack).sample().item()
        prob_attack = prob_attack.detach().cpu().numpy().squeeze()
        hidden_out = tuple(h_out.detach().cpu() for h_out in hidden_out)
        return ((action_movement, prob_movement[action_movement]),
                (action_attack, prob_attack[action_attack]),
                hidden_out)

    def train(self, batch_size=4, on_policy=False):
        s, a, r, s_, a_prob, h_in, h_out, dones, begins = [], [], [], [], [], [], [], [], []

        # sample
        for batch in self._buffer.sample(batch_size, on_policy=on_policy):
            for step in batch:
                for i, data in enumerate(step):
                    s.append(data[0])
                    a.append(data[1])
                    r.append(data[2])
                    s_.append(data[3])
                    a_prob.append(data[4])
                    h_in.append(data[5])
                    h_out.append(data[6])
                    dones.append(data[7])
                    begins.append(i == 0)

        s, a, r, s_, a_prob, dones, begins = convert_to_tensor(self._device, s, a, r, s_, a_prob, dones, begins)
        # a = a.unsqueeze(1)
        r = r.unsqueeze(1)
        # a_prob = a_prob.unsqueeze(1)
        dones = dones.unsqueeze(1)

        a_movement = a[:, 0].unsqueeze(1)
        a_movement_prob = a_prob[:, 0].unsqueeze(1)
        a_attack = a[:, 1].unsqueeze(1)
        a_attack_prob = a_prob[:, 1].unsqueeze(1)

        h_in, h_out = h_in[0], h_out[0]
        hiddens = [(h_in[0].detach().to(self._device), h_in[1].detach().to(self._device)),
                   (h_out[0].detach().to(self._device), h_out[1].detach().to(self._device))]

        q_movement, q_attack = map(lambda tensor: tensor.squeeze(1), self._model.value(s, hiddens[0]))
        q_movement_a = q_movement.gather(1, a_movement.type(torch.int64))
        q_attack_a = q_attack.gather(1, a_attack.type(torch.int64))
        pi_movement, pi_attack, _ = self._model.get_policy(s, hiddens[0])
        pi_movement = pi_movement.squeeze(1)
        pi_movement_a = pi_movement.gather(1, a_movement.type(torch.int64))
        pi_attack = pi_attack.squeeze(1)
        pi_attack_a = pi_attack.gather(1, a_attack.type(torch.int64))
        value_movement = (q_movement * pi_movement).sum(1).unsqueeze(1).detach()
        value_attack = (q_attack * pi_attack).sum(1).unsqueeze(1).detach()

        # Movement Loss
        rho_movement = pi_movement.detach() / a_movement_prob
        rho_movement_a = rho_movement.gather(1, a_movement.type(torch.int64))
        rho_movement_bar = rho_movement_a.clamp(max=self._c_sampling_ratio)
        correction_coeff_movement = (1 - self._c_sampling_ratio / rho_movement).clamp(min=0)

        q_retrace_movement = value_movement[-1] * dones[-1]
        q_retraces_movement = []
        for i in reversed(range(len(r))):
            q_retrace_movement = r[i] + self._gamma * q_retrace_movement
            q_retraces_movement.append(q_retraces_movement.item())
            q_retrace_movement = rho_movement_bar[i] * (q_retrace_movement - q_movement_a[i]) + value_movement[i]
            if begins[i] and i != 0:
                q_retrace_movement = value_movement[i-1] * dones[i-1]
        q_retraces_movement.reverse()
        q_retraces_movement = torch.tensor(q_retraces_movement).unsqueeze(1).to(self._device)

        loss_movement_1 = -rho_movement_bar * torch.log(pi_movement_a) * (q_retraces_movement - value_movement)
        loss_movement_2 = -correction_coeff_movement * pi_movement.detach() * torch.log(pi_movement) * (q_movement.detach() - value_movement)
        loss_movement = loss_movement_1 + loss_movement_2.sum(1) + F.smooth_l1_loss(q_movement_a, q_retraces_movement)

        # Attack Loss
        rho_attack = pi_attack.detach() / a_attack_prob
        rho_attack_a = rho_attack.gather(1, a_attack.type(torch.int64))
        rho_attack_bar = rho_attack_a.clamp(max=self._c_sampling_ratio)
        correction_coeff_attack = (1 - self._c_sampling_ratio / rho_attack).clamp(min=0)

        q_retrace_attack = value_attack[-1] * dones[-1]
        q_retraces_attack = []
        for i in reversed(range(len(r))):
            q_retrace_attack = r[i] + self._gamma * q_retrace_attack
            q_retraces_attack.append(q_retraces_attack.item())
            q_retrace_attack = rho_attack_bar[i] * (q_retrace_attack - q_attack_a[i]) + value_attack[i]
            if begins[i] and i != 0:
                q_retrace_attack = value_attack[i-1] * dones[i-1]
        q_retraces_attack.reverse()
        q_retraces_attack = torch.tensor(q_retraces_attack).unsqueeze(1).to(self._device)

        loss_attack_1 = -rho_attack_bar * torch.log(pi_attack_a) * (q_retraces_attack - value_attack)
        loss_attack_2 = -correction_coeff_attack * pi_attack.detach() * torch.log(pi_attack) * (q_attack.detach() - value_attack)
        loss_attack = loss_attack_1 + loss_attack_2.sum(1) + F.smooth_l1_loss(q_attack_a, q_retraces_attack)

        # Total Loss
        loss = loss_movement + loss_attack
        loss_value = loss.mean().item()

        self._optim.zero_grad()
        loss.mean.backward()
        self._optim.step()

        return loss_value

    def state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    @property
    def buffer(self):
        return self._buffer

    @property
    def rnn_output_size(self):
        return self._model.rnn_output_size


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
        s, a, r, s_, a_prob, h_in, h_out, dones, begins = [], [], [], [], [], [], [], [], []

        # sample
        for batch in self._buffer.sample(batch_size, on_policy=on_policy):
            for step in batch:
                for i, data in enumerate(step):
                    s.append(data[0])
                    a.append(data[1])
                    r.append(data[2])
                    s_.append(data[3])
                    a_prob.append(data[4])
                    h_in.append(data[5])
                    h_out.append(data[6])
                    dones.append(data[7])
                    begins.append(i == 0)

        s, a, r, s_, a_prob, dones, begins = convert_to_tensor(self._device, s, a, r, s_, a_prob, dones, begins)
        a = a.unsqueeze(1)
        r = r.unsqueeze(1)
        a_prob = a_prob.unsqueeze(1)
        dones = dones.unsqueeze(1)
        h_in, h_out = h_in[0], h_out[0]
        hiddens = [(h_in[0].detach().to(self._device), h_in[1].detach().to(self._device)),
                   (h_out[0].detach().to(self._device), h_out[1].detach().to(self._device))]

        q = self._model.value(s, hiddens[0]).squeeze(1)
        q_a = q.gather(1, a.type(torch.int64))
        pi, _ = self._model.get_policy(s, hiddens[0], training=True)
        pi = pi.squeeze(1)
        pi_a = pi.gather(1, a.type(torch.int64))
        v = (q * pi).sum(1).unsqueeze(1).detach()

        rho = pi.detach() / a_prob
        rho_a = rho.gather(1, a.type(torch.int64))
        rho_bar = rho_a.clamp(max=self._c_sampling_ratio)
        correction_coeff = (1 - self._c_sampling_ratio / rho).clamp(min=0)

        q_retrace = v[-1] * dones[-1]
        q_retraces = []
        for i in reversed(range(len(r))):
            q_retrace = r[i] + self._gamma * q_retrace
            q_retraces.append(q_retrace.item())
            q_retrace = rho_bar[i] * (q_retrace - q_a[i]) + v[i]

            if begins[i] and i != 0:
                q_retrace = v[i-1] * dones[i-1]     # when a new sequence begins

        q_retraces.reverse()
        q_retrace = torch.FloatTensor(q_retraces).unsqueeze(1).to(self._device)

        loss1 = -rho_bar * torch.log(pi_a) * (q_retrace - v)
        loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach() - v)  # bias correction term
        loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_retrace)

        loss_value = loss.mean().item()

        self._optim.zero_grad()
        loss.mean().backward()
        self._optim.step()

        return loss_value

    def state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    @property
    def buffer(self):
        return self._buffer

    @property
    def rnn_output_size(self):
        return self._model.rnn_output_size


if __name__ == "__main__":
    pass
