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
        self.attention_layer = MultiHeadAttention(shape[-1])
        # self.attention_layer = nn.MultiheadAttention(shape[-1], num_heads=2)
        self.linear2 = nn.Linear(shape[-1], 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 512)
        self.actor = nn.Linear(512, ac_s)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        # x = x.unsqueeze(1)
        # attn, attn_w = self.attention_layer(x, x, x)
        attn, attn_w = self.attention_layer(x, x, x)
        # attn_w = F.softmax(attn, dim=1)
        y = F.relu(self.linear2(attn))
        y = F.relu(self.linear3(y))
        y = F.relu(self.linear4(y))
        actor_logits = self.actor(y)
        values = self.critic(y)
        prob = F.softmax(actor_logits, dim=-1)
        action = Categorical(prob).sample()
        return actor_logits, values, action, attn_w.detach().numpy()


class MultiHeadAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w_qs = nn.Linear(size, size)
        self.w_ks = nn.Linear(size, size)
        self.w_vs = nn.Linear(size, size)

        self.attention = ScaledDotProductAttention()    # TODO: Additive

    def forward(self, q, k, v):
        residual = q
        q = self.w_qs(q)    # .permute(0, 2, 3, 1)
        k = self.w_ks(k)    # .permute(0, 2, 3, 1)
        v = self.w_vs(v)    # .permute(0, 2, 3, 1)
        attention, attn_w = self.attention(q, k, v)
        out = attention + residual
        return out, attn_w


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-2, -1))
        output = torch.matmul(F.softmax(attn / k.shape[-1], dim=1), v)
        # output = torch.matmul(attn / k.shape[-1], v)
        attn_w = F.softmax(output, dim=1)
        return output, attn_w


"""
class MultiHeadAttention_(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        print(f'Attn: {q.shape} {k.shape} {v.shape}')
        attn_w = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)),
            dim=1
        )
        attn = torch.bmm(attn_w.unsqueeze(0), v.unsqueeze(0))   # unsqueeze(0)
        print(f'Attn: {attn.shape}')
        return attn, attn_w
"""


def main():
    pass


if __name__ == "__main__":
    # main()

    input_shape = (84,)
    output_shape = 10
    batch_size = 1

    inputs = np.random.uniform(0.0, 1.0, (batch_size, *input_shape))
    inputs = torch.from_numpy(inputs).float()

    model = ActorCriticNet(input_shape, ac_s=output_shape)
    logits, values, action, attn_w = model(inputs)
    print('logits.shape:', logits.shape)
    print('values.shape:', values.shape)
    print('attn_w.shape:', attn_w.shape)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_w.T, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''])
    #ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                   ['<EOS>'], rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
