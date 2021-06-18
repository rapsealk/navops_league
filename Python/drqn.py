#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import time
from datetime import datetime
from itertools import count

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from drqn_utils import EpisodeMemory, EpisodeBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--buffer-size', type=int, default=20000)
parser.add_argument('--time-horizon', type=int, default=200)
parser.add_argument('--sequence-length', type=int, default=128)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--target-update-period', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--tau', type=float, default=1e-2)
args = parser.parse_args()


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

    def forward(self, x):
        pass


class DRQN(nn.Module):

    def __init__(self, input_size, action_size=18, hidden_size=512, rnn_hidden_size=64):
        super(DRQN, self).__init__()

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._hidden_size = hidden_size
        self._rnn_hidden_size = rnn_hidden_size
        self._action_size = action_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, self.rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(self.rnn_hidden_size, action_size)

    def forward(self, x, h_in):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x, h_out = self.lstm(x, h_in)
        x = self.fc(x)
        return x, h_out

    def get_action(self, x, h_in, epsilon=0.0):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.view(-1, 1, x.shape[-1]).to(self.device)
        logits, h_out = self.__call__(x, h_in)
        if np.random.random() > epsilon:
            action = F.softmax(logits, dim=-1)
            action = torch.argmax(action, dim=-1).squeeze().item()
        else:
            action = np.random.randint(self.action_size)
        return action, logits.detach(), h_out

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
    def action_size(self):
        return self._action_size

    @property
    def device(self):
        return self._device


def discount_rewards(rewards, gamma=args.gamma):
    rewards_ = np.zeros_like(rewards)
    rewards_[0] = rewards[-1]
    for i in range(1, len(rewards)):
        rewards_[i] = rewards_[i-1] * gamma + rewards[-i-1]
    return rewards_[::-1]


def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None,
          optimizer=None,
          learning_rate=1e-3,
          gamma=0.99,
          batch_size=args.batch_size):

    assert device is not None, "None Device input: device should be selected."

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

    observations = torch.FloatTensor(observations.reshape(batch_size, sequence_length, -1)).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size, sequence_length, -1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size, sequence_length, -1)).to(device)
    next_observations = torch.FloatTensor(next_observations.reshape(batch_size, sequence_length, -1)).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size, sequence_length, -1)).to(device)

    h_in = target_q_net.reset_hidden_state(batch_size=batch_size)
    h_in = tuple([h.to(device) for h in h_in])

    q_target, _ = target_q_net(next_observations, h_in)

    q_target_max = q_target.max(2)[0].view(batch_size, sequence_length, -1).detach()
    targets = rewards + gamma*q_target_max*dones

    h_in = q_net.reset_hidden_state(batch_size=batch_size)
    h_in = tuple([h.to(device) for h in h_in])
    q_out, _ = q_net(observations, h_in)
    q_a = q_out.gather(dim=2, index=actions)

    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    env = gym.make('MountainCar-v0')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_model = DRQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()     # no_grad

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    memory = EpisodeMemory(max_epi_num=args.buffer_size,
                           max_epi_len=args.time_horizon,
                           batch_size=args.batch_size,
                           lookup_step=args.sequence_length,
                           random_update=True)

    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.01

    train_step = 0

    time_ = time.time()

    for episode in count(1):

        state = env.reset()
        h_in = model.reset_hidden_state(batch_size=1)

        episode_buffer = EpisodeBuffer()
        rewards = []

        for t in count(1):
            env.render()

            action, logits, h_in = model.get_action(state, h_in, epsilon=epsilon)

            next_state, reward, done, info = env.step(action)
            reward += np.power((next_state[0] + 0.5) * 10, 2) / 10 + next_state[0]
            if next_state[0] >= 0.5:
                reward = 10
            episode_buffer.put(state, action, reward, next_state, done)

            # print(f'Ep:{episode}:{t} state: {state}, action: {action} reward: {reward}, next_state: {next_state}, done: {done}')
            # print(f'Ep:{episode}:{t} state: {state}')

            state = next_state
            rewards.append(reward)

            if len(memory) >= args.batch_size and (t+1) % args.target_update_period == 0:
                loss = train(model, target_model, memory,
                             device, optimizer,
                             args.learning_rate, args.gamma)
                # print(f'Loss: {loss}')
                if (train_step := train_step + 1) % 20 == 0:
                    target_model.load_state_dict(model.state_dict())

            if done:
                t_ = time.time() - time_
                print(f'[{datetime.now().isoformat()}] ({int(t_//3600):02d}h {int(t_%3600//60):02d}m {(t_%3600)%60:02.2f}s) Ep:{episode}:{t} -> {np.sum(rewards)} ({info}) (epsilon={epsilon})')
                epsilon = max(epsilon * epsilon_decay, epsilon_min)
                episode_buffer.reward = discount_rewards(episode_buffer.reward)
                memory.put(episode_buffer)
                break

    env.close()


if __name__ == "__main__":
    main()
