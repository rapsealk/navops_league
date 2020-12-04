#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


class ActorCriticLSTM(nn.Module):

    def __init__(self, n=6):
        super(ActorCriticLSTM, self).__init__()

        self.recurrent = nn.LSTM(input_size=16, hidden_size=256, batch_first=True)  # (seq_len, batch, input_size)
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc_p = nn.Linear(in_features=128, out_features=128)
        self.policy = nn.Linear(in_features=128, out_features=n)
        self.fc_v = nn.Linear(in_features=128, out_features=64)
        self.value = nn.Linear(in_features=64, out_features=1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_uniform_(param.data, mode='fan_out', nonlinearity='relu')
                        # nn.init.uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, hidden_states = self.recurrent(x)
        x = self.relu(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        pi = self.relu(self.fc_p(x))
        pi = self.softmax(self.policy(pi))
        v = self.relu(self.fc_v(x))
        value = self.value(v)

        return pi, value

    def reset_hidden_states(self):
        pass


class ProximalPolicyOptimizationAgent:

    def __init__(self, n=6, model=None, gamma=0.99, lambda_=0.95, learning_rate=2e-4):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.learning_rate = learning_rate

        self.n = n
        self.model = model or ActorCriticLSTM(n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.epsilon = 0.2
        self.normalize = True

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        self.model.cpu()
        policy, _ = self.model(state.cpu())
        policy = np.squeeze(policy)
        policy = policy.detach().numpy()[-1]
        action = np.random.choice(policy.shape[-1], p=policy)
        return action

    def update(self, states, actions, next_states, rewards, dones):
        device = torch.device("cpu")

        self.model.to(device)
        policy, values = self.model(torch.from_numpy(states).float().to(device))
        _, next_values = self.model(torch.from_numpy(next_states).float().to(device))

        policy = policy.detach().to(device).numpy()
        values = values.detach().to(device).numpy()[:, -1]
        next_values = next_values.detach().to(device).numpy()[:, -1]

        advantages, target_values = self.gae(values, next_values, rewards, dones,
                                             gamma=self.gamma, lambda_=self.lambda_,
                                             normalize=self.normalize)

        self.model.to(DEVICE)

        samples = np.arange(len(states))
        np.random.shuffle(samples)

        self.optimizer.zero_grad()

        train_policy, train_value = self.model(torch.from_numpy(states).float().to(DEVICE))
        train_value = torch.squeeze(train_value)
        advantages = torch.from_numpy(advantages).float().to(DEVICE)
        target_values = torch.from_numpy(target_values).float().to(DEVICE)
        actions = torch.from_numpy(actions).float().to(DEVICE)
        old_policy = torch.from_numpy(policy).float().to(DEVICE)

        train_policy = train_policy[:, -1]
        old_policy = old_policy[:, -1]

        entropy = torch.mean(-train_policy * torch.log(train_policy + 1e-8)) * 0.1
        prob = torch.sum(train_policy * actions, axis=1)
        old_prob = torch.sum(old_policy * actions, axis=1)
        log_pi = torch.log(prob + 1e-8)
        log_old_pi = torch.log(old_prob + 1e-8)

        ratio = torch.exp(log_pi - log_old_pi)
        clipped_ratio = torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon)
        minimum = torch.min(torch.mul(advantages, clipped_ratio), torch.mul(advantages, ratio))
        loss_pi = -torch.mean(minimum) + entropy
        loss_value = torch.mean(torch.square(target_values - train_value))
        total_loss = loss_pi + loss_value

        loss = total_loss.item()
        total_loss.backward()
        self.optimizer.step()

        return loss

    def gae(self, values, next_values, rewards, dones,
            gamma=0.99, lambda_=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * v_ - v
                  for r, v, v_, d in zip(rewards, values, next_values, dones)]
        deltas = np.stack(deltas)

        gaes = np.copy(deltas)
        for t in reversed(range(deltas.shape[0]-1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lambda_ * gaes[t+1]
        target_values = gaes + values

        if normalize:
            gaes = (gaes - np.mean(gaes)) / (np.std(gaes) + 1e-8)

        return gaes, target_values

    def save(self, path=os.path.join(os.path.dirname(__file__), 'checkpoint.pth.tar')):
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path=os.path.join(os.path.dirname(__file__), 'checkpoint.pth.tar')):
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        try:
            self.model.load_state_dict(weights)
        except:
            pass


if __name__ == "__main__":
    model = ActorCriticLSTM()
    agent = ProximalPolicyOptimizationAgent()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    """
    print('inputs:', inputs.dtype, inputs.shape)
    pi, value = model(inputs)
    print('pi:', np.sum(pi[0, 0, :].detach().cpu().numpy()))
    print('value:', value.shape)
    """

    inputs = np.random.uniform(-1, 1, (64, 1, 16))
    action = agent.get_action(inputs)
    print('action:', action)

    inputs = torch.from_numpy(inputs).float()   # .to(device)
    pi, value = model(inputs)
    print('pi:', pi.shape)
    print('value:', value.shape)
