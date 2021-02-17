#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def convert_to_tensor(device, *args):
    return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 256)
        self.linear5 = nn.Linear(256, output_size)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x


class DQNAgent:

    def __init__(self, input_size, output_size, batch_size=1024, force_cpu=False):
        if force_cpu:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = Model(input_size, output_size).to(self._device)
        self._optim = optim.RMSprop(self._model.parameters())
        self._batch_size = batch_size

    def get_action(self, inputs):
        action = self._model(torch.FloatTensor(inputs).to(self._device))
        return torch.argmax(action).detach().cpu().numpy()

    def train(self, memory):
        s, a, r, s_, dones = self._make_batch(memory, self._batch_size)
        y_j = self._model(s).gather(dim=1, index=a.type(torch.int64).unsqueeze(1))
        loss = F.smooth_l1_loss(y_j, r.unsqueeze(1))

        self._optim.zero_grad()
        loss.backward()
        for param in self._model.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optim.step()

        return loss.detach().cpu().numpy()

    def _make_batch(self, memory, batch_size=1024):
        s, a, r, s_, dones = memory.sample(batch_size)
        s, a, r, s_, dones = convert_to_tensor(self._device, s, a, r, s_, dones)
        return s, a, r, s_, dones

    def state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def save(self, path: str):
        torch.save({
            "params": self._model.parameters(),
            "optim": self._optim.parameters(),
            # TODO: epsilon
        }, path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self._model.load(state_dict["params"])
        self._optim.load(state_dict["optim"])


class LstmActorCritic(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        action_space=None,
        num_layers=256,
        batch_size=256
    ):
        super(LstmActorCritic, self).__init__()

        # Actor
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_size)
        self.log_std = nn.Linear(hidden_size, output_size)

        # Critic
        self.q1 = nn.Sequential(
            nn.Linear(input_size + output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_size + output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(weights_init_)

        self._num_layers = num_layers
        self._batch_size = batch_size
        self._hidden_size = hidden_size

        self._hidden = (torch.randn((num_layers, batch_size, hidden_size)),
                        torch.randn((num_layers, batch_size, hidden_size)))

        # FIXME: action rescaling
        if action_space is None:
            self._action_scale = torch.tensor(1.0)
            self._action_bias = torch.tensor(0.0)
        else:
            self._action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2)
            self._action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2)

    def forward(self, inputs):
        x, self._hidden = self.lstm(inputs, self._hidden)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self._action_scale + self._action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(self._action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self._action_scale + self._action_bias

        state_action = torch.cat([inputs, action], axis=2)
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)

        return action, log_prob, mean, (q1, q2)

    def reset(self):
        del self._hidden
        self._hidden = (torch.randn((self._num_layers, self._batch_size, self._hidden_size)),
                        torch.randn((self._num_layers, self._batch_size, self._hidden_size)))

    def to(self, device):
        self._hidden = tuple(hidden.to(device) for hidden in self._hidden)
        self._action_scale = self._action_scale.to(device)
        self._action_bias = self._action_bias.to(device)
        return super(LstmActorCritic, self).to(device)


class SoftActorCriticAgent:

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        action_space,
        num_layers,
        batch_size,
        gamma=0.99,
        tau=0.05,
        alpha=0.2,
        learning_rate=3e-4,
        force_cpu=False
    ):
        self._batch_size = batch_size

        self._gamma = gamma
        self._tau = tau
        self._alpha = alpha
        self._learning_rate = learning_rate

        self._target_update_interval = 1
        self._automatic_entropy_tuning = True

        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = LstmActorCritic(
            input_size,
            output_size,
            hidden_size,
            action_space,
            num_layers,
            batch_size
        ).to(self.device)
        self._optim = optim.Adam(self._model.parameters(), lr=learning_rate)
        self._critic_target = LstmActorCritic(
            input_size,
            output_size,
            hidden_size,
            action_space,
            num_layers,
            batch_size
        ).to(self.device)
        for target_param, param in zip(self._critic_target.parameters(), self._model.parameters()):
            target_param.data.copy_(param.data)

        # Gaussian
        if self._automatic_entropy_tuning:
            self._target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self._alpha_optim = optim.Adam([self._log_alpha], lr=learning_rate)

    def reset(self):
        self._model.reset()

    def get_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        if evaluate:
            _, _, action, _ = self._model(state)
        else:
            action, _, _, _ = self._model(state)
        del state
        return action.detach().cpu().numpy()

    def descent(self, gradient, worker=None):
        worker.optim.zero_grad()
        self._optim.zero_grad()
        gradient.backward()
        for local_param, global_param in zip(worker.model.parameters(), self._model.parameters()):
            global_param._grad = local_param.grad.to(self.device)
        self._optim.step()
        del gradient
        torch.cuda.empty_cache()

    def compute_gradient(self, memory, batch_size, updates=0):
        s, a, r, s_, dones = memory.sample(batch_size)
        s, a, r, s_, dones = convert_to_tensor(self.device, s, a, r, s_, dones)
        r = r.unsqueeze(1)
        dones = dones.unsqueeze(1)

        with torch.no_grad():
            _, next_state_log_pi, _, _ = self._model(s_)
            _, _, _, target_values = self._critic_target(s_)
            min_q_next_target = torch.min(target_values[0].detach(), target_values[1].detach()) - self._alpha * next_state_log_pi.detach()
            min_q_next_target = min_q_next_target[:, -1, -1:]
            next_q_value = r + dones * self._gamma * min_q_next_target
            next_q_value = next_q_value.detach()

        pi, log_pi, _, (qf1, qf2) = self._model(s)
        qf1 = qf1[:, -1]
        qf2 = qf2[:, -1]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        pi_loss = (self._alpha * log_pi - torch.min(qf1, qf2)).mean()

        loss = qf_loss + pi_loss
        del s, a, r, s_, dones
        return loss, qf_loss.detach().item(), pi_loss.detach().item()

        """
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return qf1_loss.item(), qf2_loss.item(), pi_loss.item()
        """

        return loss

    def get_state_dict(self):
        return self._model.state_dict()

    def set_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def share_memory(self):
        self._model.share_memory()

    @property
    def model(self):
        return self._model

    @property
    def optim(self):
        return self._optim


if __name__ == "__main__":
    import gym
    import gym_rimpac   # noqa: F401

    env = gym.make('RimpacDiscrete-v0', mock=True)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    inputs = np.random.uniform(-1.0, 1.0, env.observation_space.shape)
    action = agent.get_action(inputs)

    print('inputs:', inputs)
    print('action:', action)
