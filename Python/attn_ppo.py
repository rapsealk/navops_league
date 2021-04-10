#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


# https://github.com/RvuvuzelaM/self-attention-ppo-pytorch
class ActorCriticNet(nn.Module):
    def __init__(self, shape, ac_s):
        super().__init__()
        self.linear1 = nn.Linear(shape[-1], out_features=32)
        self.attention_layer = MultiHeadAttention(32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 512)
        self.actor = nn.Linear(512, ac_s)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        attn = self.attention_layer(y, y, y)
        attn_w = F.softmax(attn, dim=1).detach().numpy()
        y = F.relu(self.linear2(attn))
        y = F.relu(self.linear3(y))
        y = F.relu(self.linear4(y))
        actor_logits = self.actor(y)
        values = self.critic(y)
        prob = F.softmax(actor_logits, dim=-1)
        action = Categorical(prob).sample()
        return actor_logits, values, action, attn_w


class MultiHeadAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w_qs = nn.Linear(size, size)
        self.w_ks = nn.Linear(size, size)
        self.w_vs = nn.Linear(size, size)

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v):
        residual = q
        q = self.w_qs(q)    # .permute(0, 2, 3, 1)
        k = self.w_ks(k)    # .permute(0, 2, 3, 1)
        v = self.w_vs(v)    # .permute(0, 2, 3, 1)

        attention = self.attention(q, k, v) # .permute(0, 3, 1, 2)

        out = attention + residual
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-2, -1))
        output = torch.matmul(attn, v)
        return output


def main():
    pass


if __name__ == "__main__":
    # main()

    input_shape = (84,)
    inputs = np.random.uniform(0.0, 1.0, (4, *input_shape))
    inputs = torch.from_numpy(inputs).float()

    model = ActorCriticNet(input_shape, ac_s=3)
    logits, values, action, attn_w = model(inputs)
    print('attn_w.shape:', attn_w.shape)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_w.T, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                   ['<EOS>'], rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
