#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

autograd.set_detect_anomaly(True)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def convert_to_tensor(device, *args):
    return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))


"""
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
"""

"""
class ActorCriticModel(nn.Module):

    def __init__(
        self,
        inputs_size,
        outputs_size,
        hidden_size=256,
        batch_size=256,
        num_layers=256
    ):
        super(ActorCriticModel, self).__init__()
        self.lstm = nn.LSTM(inputs_size, hidden_size, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.head = nn.Linear(hidden_size, outputs_size)

        self.q1_fc1 = nn.Linear(inputs_size + outputs_size, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_fc3 = nn.Linear(hidden_size, 1)
        self.q2_fc1 = nn.Linear(inputs_size + outputs_size, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_fc3 = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)

        self.hidden = (torch.randn((num_layers, batch_size, hidden_size)),
                       torch.randn((num_layers, batch_size, hidden_size)))

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = self.head(x)

        x = torch.cat([x, pi], dim=1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return pi, q1, q2
"""


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(x))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicyNetwork(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        num_layers=256,
        batch_size=256
    ):
        super(GaussianPolicyNetwork, self).__init__()

        self.lstm = nn.LSTM(num_inputs, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.hidden = (torch.randn((num_layers, batch_size, hidden_dim)),
                       torch.randn((num_layers, batch_size, hidden_dim)))

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2)

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyNetwork, self).to(device)


class SoftActorCriticAgent:

    def __init__(
        self,
        num_inputs,
        action_space,
        hidden_dim=256,
        num_layers=256,
        batch_size=256,
        gamma=0.99,
        tau=0.05,
        alpha=0.2,
        lr=0.0003
    ):
        self._batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self.target_update_interval = 1
        self.automatic_entropy_tuning = True    # False
        # self.replay_size = 1e6

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_dim).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_dim).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Gaussian
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        self.policy = GaussianPolicyNetwork(
            num_inputs,
            action_space.shape[0],
            hidden_dim,
            action_space,
            num_layers,
            batch_size
        )
        self.policy = self.policy.to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        """Deterministic
        self.alpha = 0
        self.automatic_entropy_tuning = False
        self.policy = DeterministicPolicyNetwork(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        """

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        if not evaluate:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        # return action.detach().cpu().numpy()[-1]
        return action.detach().cpu().numpy()

    def get_trajectory_batch(self, buffer, batch_size):
        s, a, r, s_, dones = buffer.sample(batch_size)
        s, a, r, s_, dones = convert_to_tensor(self.device, s, a, r, s_, dones)
        r = r.unsqueeze(1)
        dones = dones.unsqueeze(1)
        return s, a, r, s_, dones

    def get_value_gradient(self, s, a, r, s_, dones):
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(s_)
            s_a = torch.cat([s_, next_state_action.to(self.device)], axis=2)
            qf1_next_target, qf2_next_target = self.critic_target(s_a)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target = min_qf_next_target[:, -1, -1:]
            next_q_value = r + dones * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(torch.cat([s, a], dim=2))
        qf1_loss = F.mse_loss(qf1[:, -1], next_q_value)
        qf2_loss = F.mse_loss(qf2[:, -1], next_q_value)
        qf_loss = qf1_loss + qf2_loss

        return qf_loss

    def get_policy_gradient(self, s, a, r, s_, dones):
        # Value Loss

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(s_)
            print(f'cat - s_: {s_.shape}, next_state_action: {next_state_action.shape}')
            s_a = torch.cat([s_, next_state_action.to(self.device)], axis=2)
            qf1_next_target, qf2_next_target = self.critic_target(s_a)
            # qf1_next_target, qf2_next_target = self.critic_target(s_, next_state_action.to(self.device))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # min_qf_next_target = min_qf_next_target[-1, -1]
            min_qf_next_target = min_qf_next_target[:, -1, -1:]
            print('min_qf_next_target:', min_qf_next_target.shape)
            next_q_value = r + dones * self.gamma * min_qf_next_target

        print(f'critic - s: {s.shape}, a: {a.shape}')
        qf1, qf2 = self.critic(torch.cat([s, a], dim=2))
        qf1 = qf1[:, -1]
        qf2 = qf2[:, -1]
        print(f'qf1: {qf1.shape}, qf2: {qf2.shape}, next_q_value: {next_q_value.shape}')
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        pi, log_pi, _ = self.policy.sample(s)

        # Policy Loss

        qf1_pi, qf2_pi = self.critic(torch.cat([s, pi], dim=2))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        else:
            alpha_loss = None

        return policy_loss

    def compute_gradient(self, buffer, batch_size, updates):
        s, a, r, s_, dones = buffer.sample(batch_size)
        s, a, r, s_, dones = convert_to_tensor(self.device, s, a, r, s_, dones)
        r = r.unsqueeze(1)
        dones = dones.unsqueeze(1)
        print('compute_gradient')
        print(f's: {s.shape}, a: {a.shape}, r: {r.shape}, s_: {s_.shape}, dones: {dones.shape}')

        # Value Loss

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(s_)
            print(f'cat - s_: {s_.shape}, next_state_action: {next_state_action.shape}')
            s_a = torch.cat([s_, next_state_action.to(self.device)], axis=2)
            qf1_next_target, qf2_next_target = self.critic_target(s_a)
            # qf1_next_target, qf2_next_target = self.critic_target(s_, next_state_action.to(self.device))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # min_qf_next_target = min_qf_next_target[-1, -1]
            min_qf_next_target = min_qf_next_target[:, -1, -1:]
            print('min_qf_next_target:', min_qf_next_target.shape)
            next_q_value = r + dones * self.gamma * min_qf_next_target

        print(f'critic - s: {s.shape}, a: {a.shape}')
        qf1, qf2 = self.critic(torch.cat([s, a], dim=2))
        qf1 = qf1[:, -1]
        qf2 = qf2[:, -1]
        print(f'qf1: {qf1.shape}, qf2: {qf2.shape}, next_q_value: {next_q_value.shape}')
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        pi, log_pi, _ = self.policy.sample(s)

        # Policy Loss

        qf1_pi, qf2_pi = self.critic(torch.cat([s, pi], dim=2))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        else:
            alpha_loss = None

        return qf_loss, policy_loss, alpha_loss

    def descent_gradient(self, local, q_loss=None, pi_loss=None, alpha_loss=None):
        if q_loss is not None:
            self.critic_optim.zero_grad()
            q_loss.backward()
            for local_param, global_param in zip(local.critic.parameters(), self.critic.parameters()):
                global_param._grad = local_param.grad
            self.critic_optim.step()
        elif pi_loss is not None:
            self.policy_optim.zero_grad()
            pi_loss.backward()
            for local_param, global_param in zip(local.policy.parameters(), self.policy.parameters()):
                global_param._grad = local_param.grad
            self.policy_optim.step()
        """
        if self.automatic_entropy_tuning:
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha._grad = local.log_alpha.grad
            self.alpha_optim.step()
        """

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        states, actions, rewards, next_states, masks = memory.sample(batch_size=batch_size)

        # states, actions, rewards, next_states, masks \
        #     = convert_to_tensor(self.device, states, actions, rewards, next_states, masks)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
            # print(f'cat - s_: {s_.shape}, next_state_action: {next_state_action.shape}')
            s_a = torch.cat([next_states, next_state_action.to(self.device)], axis=2)
            qf1_next_target, qf2_next_target = self.critic_target(s_a)
            # qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action.to(self.device))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + masks * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(torch.cat([states, actions], dim=2)) # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(states)

        qf1_pi, qf2_pi = self.critic(torch.cat([states, pi], dim=2))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()    # Tensorboard
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

        """
        # Save model parameters
        def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
            if not os.path.exists('models/'):
                os.makedirs('models/')

            if actor_path is None:
                actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
            if critic_path is None:
                critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
            print('Saving models to {} and {}'.format(actor_path, critic_path))
            torch.save(self.policy.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)

        # Load model parameters
        def load_model(self, actor_path, critic_path):
            print('Loading models from {} and {}'.format(actor_path, critic_path))
            if actor_path is not None:
                self.policy.load_state_dict(torch.load(actor_path))
            if critic_path is not None:
                self.critic.load_state_dict(torch.load(critic_path))
        """

    def get_state_dict(self):
        return self.policy.state_dict(), self.critic.state_dict()

    def set_state_dict(self, state_dicts):
        self.policy.load_state_dict(state_dicts[0])
        self.critic.load_state_dict(state_dicts[1])

    def save(self, path: str):
        if not path.endswith('.ckpt'):
            path += '.ckpt'
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        torch.save({
            # "cuda": torch.cuda.is_available(),
            "state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.policy_optim.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            # ...
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": self.alpha,
            "lr": self.lr,

            "target_update_interval": self.target_update_interval,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "target_entropy": (self.automatic_entropy_tuning or None) and self.target_entropy,
            "log_alpha": (self.automatic_entropy_tuning or None) and self.log_alpha,    # .item()
            "alpha_optim": (self.automatic_entropy_tuning or None) and self.alpha_optim.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["state_dict"])
        self.policy_optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        # ...
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.alpha = checkpoint["alpha"]
        self.lr = checkpoint["lr"]

        self.target_update_interval = checkpoint["target_update_interval"]
        self.automatic_entropy_tuning = checkpoint["automatic_entropy_tuning"]
        if self.automatic_entropy_tuning:
            self.target_entropy = checkpoint["target_entropy"]
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optim.load_state_dict(checkpoint["alpha_optim"])


if __name__ == "__main__":
    pass
