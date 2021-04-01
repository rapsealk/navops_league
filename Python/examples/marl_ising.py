#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt


def init_grid(size=(10,)):
    grid = torch.randn(*size)
    grid[grid > 0] = 1
    grid[grid <= 0] = 0
    grid = grid.byte()
    return grid


def get_reward(s, a):
    r = -1
    for i in s:
        if i == a:
            r += 0.9
    r *= 2
    return r


def gen_params(n, size):
    ret = []
    for i in range(n):
        vec = torch.randn(size) / 10.0
        vec.requires_grad = True
        ret.append(vec)
    return ret


def qfunc(s, theta, layers=[(4, 20), (20, 2)], afn=torch.tanh):
    l1n = layers[0]
    l1s = np.prod(l1n)
    theta_1 = theta[0:l1s].reshape(l1n)
    l2n = layers[1]
    l2s = np.prod(l2n)
    theta_2 = theta[l1s:l1s+l2s].reshape(l2n)
    bias = torch.ones((1, theta_1.shape[1]))
    l1 = s @ theta_1 + bias
    l1 = torch.nn.functional.elu(l1)
    l2 = afn(l1 @ theta_2)
    return l2.flatten()


def get_substate(b):
    s = torch.zeros(2)
    if b > 0:
        s[1] = 1
    else:
        s[0] = 1
    return s


def joint_state(s):
    s1 = get_substate(s[0])
    s2 = get_substate(s[1])
    return (s1.reshape(2, 1) @ s2.reshape(1, 2)).flatten()


def main():
    plt.figure(figsize=(8, 5))
    size = (20,)
    hidden_layer = 20
    params = gen_params(size[0], 4 * hidden_layer + hidden_layer * 2)
    grid = init_grid(size=size)
    grid_ = grid.clone()
    print(grid)
    plt.imsave('ising1.png', np.expand_dims(grid, axis=0))

    epochs = 1000
    lr = 0.0001
    losses = [[] for _ in range(size[0])]
    for i in range(epochs):
        for j in range(size[0]):
            l = j - 1 if j - 1 >= 0 else size[0] - 1
            r = j + 1 if j + 1 < size[0] else 0
            state_ = grid[[l, r]]
            state = joint_state(state_)
            qvalues = qfunc(state.float().detach(), params[j], layers=[(4, hidden_layer), (hidden_layer, 2)])
            qmax = torch.argmax(qvalues, dim=0).detach().item()
            action = int(qmax)
            reward = get_reward(state_.detach(), action)
            with torch.no_grad():
                target = qvalues.clone()
                target[action] = reward
            loss = torch.sum(torch.pow(qvalues - target, 2))
            losses[j].append(loss.detach().numpy())
            loss.backward()
            with torch.no_grad():
                params[j] = params[j] - lr * params[j].grad
            params[j].requires_grad = True
        with torch.no_grad():
            grid.data = grid_.data

        total_loss = 0
        for k in range(size[0]):
            total_loss += losses[k][-1]
        print(f'Epoch #{i} - Loss: {total_loss}')

    plt.imsave('ising1.png', np.expand_dims(grid, axis=0))


if __name__ == "__main__":
    main()
