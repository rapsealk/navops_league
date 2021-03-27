#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random
from collections import deque
from functools import reduce
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym  # noqa: F401
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from skimage.transform import resize

params = {
    "batch_size": 150,
    "beta": 0.2,
    "lambda": 0.1,
    "eta": 1.0,
    "gamma": 0.2,
    "max_episode_length": 100,
    "min_progress": 15,
    "action_repeats": 6,
    "frames_per_state": 3
}


class ExperienceReplay:

    def __init__(self, n=500, batch_size=100):
        self.n = n
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0

    def append(self, *args):
        self.counter += 1
        if self.counter % 500 == 0:
            self.shuffle_memory()
        if len(self.memory) < self.n:
            self.memory.append(args)    # (state1, action, reward, state2)
        else:
            rand_index = np.random.randint(0, self.n-1)
            self.memory[rand_index] = args

    def shuffle_memory(self):
        random.shuffle(self.memory)

    def get_batch(self):
        batch_size = min(len(self.memory), self.batch_size)
        if len(self.memory) < 1:
            return None

        indices = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        state1_batch = torch.stack([x[0].squeeze(0) for x in batch], dim=0)
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(0) for x in batch], dim=0)
        return state1_batch, action_batch, reward_batch, state2_batch


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), 2, 1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), 2, 1)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), 2, 1)
        self.linear = nn.Linear(288, 100)
        self.linear2 = nn.Linear(100, 12)

    def forward(self, x):
        y = F.normalize(x)
        y = F.elu(self.conv1(y))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim=2)
        y = y.view((y.shape[0], -1, 32))
        y = y.flatten(start_dim=1)
        y = F.elu(self.linear(y))
        y = self.linear2(y)
        return y


class ICMEncoderModule(nn.Module):

    def __init__(self):
        super(ICMEncoderModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), 2, 1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), 2, 1)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), 2, 1)

    def forward(self, x):
        y = F.normalize(x)
        y = F.elu(self.conv1(y))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim=1)
        return y


class ICMForwardModule(nn.Module):

    def __init__(self):
        super(ICMForwardModule, self).__init__()

        self.linear1 = nn.Linear(300, 256)
        self.linear2 = nn.Linear(256, 288)

    def forward(self, state, action):
        action_ = torch.zeros((action.shape[0], 12))
        indices = torch.stack((torch.arange(action.shape[0]), action.squeeze()), dim=0).tolist()
        action_[indices] = 1.0
        x = torch.cat((state, action_), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y


class ICMInverseModule(nn.Module):

    def __init__(self):
        super(ICMInverseModule, self).__init__()

        self.linear1 = nn.Linear(576, 256)
        self.linear2 = nn.Linear(256, 12)

    def forward(self, state1, state2):
        x = torch.cat((state1, state2), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y, dim=1)
        return y


def downscale_observation(obs, new_size=(42, 42), to_gray=True):
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    return resize(obs, new_size, anti_aliasing=True)


def prepare_state(state):
    return torch.from_numpy(downscale_observation(state, to_gray=True)).float().unsqueeze(0)


def prepare_multi_state(state1, state2):
    state1 = state1.clone()
    state1[0][0] = state1[0][1]
    state1[0][1] = state2[0][2]
    state1[0][2] = torch.from_numpy(downscale_observation(state2, to_gray=True)).float()
    return state1


def prepare_initial_state(state, n=3):
    state_ = torch.from_numpy(downscale_observation(state, to_gray=True)).float()
    return state_.repeat((n, 1, 1)).unsqueeze(0)


def policy(qvalues, eps=None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0, high=7, size=(1,))
        return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues)), num_samples=1)


def loss_fn(q_loss, forward_loss, inverse_loss):
    loss_ = (1 - params['beta']) * inverse_loss
    loss_ += params['beta'] * forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * q_loss
    return loss


def reset_env(env):
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    return state1


def icm_prediction_loss(state1, action, state2, forward_scale=1.0, inverse_scale=1e4):
    global encoder, forward, inverse, forward_loss, inverse_loss
    state1_hat = encoder(state1)
    state2_hat = encoder(state2)
    state2_hat_pred = forward(state1_hat.detach(), action.detach())
    forward_pred_err = forward_scale * forward_loss(state2_hat_pred, state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse(state1_hat, state2_hat)
    inverse_pred_err = inverse_scale * inverse_loss(pred_action, action.detach().flatten()).unsqueeze(dim=1)
    return forward_pred_err, inverse_pred_err


def minibatch_train(use_extrinsic=True):
    global buffer, model, q_loss
    state1_batch, action_batch, reward_batch, state2_batch = buffer.get_batch()
    action_batch = action_batch.view(action_batch.shape[0], 1)
    reward_batch = reward_batch.view(reward_batch.shape[0], 1)

    forward_pred_err, inverse_pred_err = icm_prediction_loss(state1_batch, action_batch, state2_batch)
    intrinsic_reward = (1.0 / params["beta"]) * forward_pred_err
    reward = intrinsic_reward.detach()
    if use_extrinsic:
        reward += reward_batch
    q_values = model(state2_batch)
    reward += params["gamma"] * torch.max(q_values)
    reward_pred = model(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack((torch.arange(action_batch.shape[0]), action_batch.squeeze()), dim=0).tolist()
    reward_target[indices] = reward.squeeze()
    q_value_loss = 1e5 * q_loss(F.normalize(reward_pred), F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_value_loss


def main():
    global model, encoder, forward, inverse, forward_loss, inverse_loss, q_loss
    global buffer
    buffer = ExperienceReplay(n=1000, batch_size=params["batch_size"])
    model = DQN()
    encoder = ICMEncoderModule()
    forward = ICMForwardModule()
    inverse = ICMInverseModule()
    forward_loss = nn.MSELoss(reduction='none')
    inverse_loss = nn.CrossEntropyLoss(reduction='none')
    q_loss = nn.MSELoss()
    models = [model, encoder, forward, inverse]
    model_params = reduce(lambda x, y: x + y, map(lambda x: list(x.parameters()), models))
    # model_params = list(model.parameters()) \
    #              + list(encoder.parameters())
    opt = optim.Adam(lr=0.001, params=model_params)

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    ###
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    eps = 0.15
    losses = []
    episode_length = 0
    switch_to_eps_greedy = 1000
    state_deque = deque(maxlen=params["frames_per_state"])
    extrinsic_reward = 0.0
    last_x_pos = env.env.env._x_position
    ep_lengths = []

    for epoch in count(1):
        opt.zero_grad()
        episode_length += 1
        q_val_pred = model(state1)
        if epoch > switch_to_eps_greedy:
            action = int(policy(q_val_pred, eps))
        else:
            action = int(policy(q_val_pred))
        for i in range(params["action_repeats"]):
            state2, extrinsic_reward_, done, info = env.step(action)

            # env.render()

            last_x_pos = info['x_pos']
            if done:
                state1 = reset_env(env)
                break
            extrinsic_reward += extrinsic_reward_
            state_deque.append(prepare_state(state2))
        state2 = torch.stack(list(state_deque), dim=1)
        buffer.append(state1, action, extrinsic_reward, state2)
        extrinsic_reward = 0.0
        if episode_length > params["max_episode_length"]:
            if info['x_pos'] - last_x_pos < params["min_progress"]:
                done = True
            else:
                last_x_pos = info['x_pos']
        if done:
            ep_lengths.append(info['x_pos'])
            state1 = reset_env(env)
            last_x_pos = env.env.env._x_position
            episode_length = 0
        else:
            state1 = state2

        if len(buffer.memory) < params["batch_size"]:
            continue

        forward_pred_err, inverse_pred_err, q_value_loss = minibatch_train(use_extrinsic=False)
        loss = loss_fn(q_value_loss, forward_pred_err, inverse_pred_err)
        loss_list = (q_value_loss.mean(), forward_pred_err.flatten().mean(), inverse_pred_err.flatten().mean())
        losses.append(loss_list)
        loss.backward()
        opt.step()

        print(f'Epoch #{epoch}: (loss={loss.item()}, x={last_x_pos})')

    env.close()


if __name__ == "__main__":
    main()
