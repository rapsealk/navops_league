#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, kdim=None, vdim=None):
        super(MultiHeadAttention, self).__init__()
        kdim = kdim or dim
        vdim = vdim or dim

        self.w_qs = nn.Linear(dim, dim)
        self.w_ks = nn.Linear(kdim, dim)
        self.w_vs = nn.Linear(vdim, dim)

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v):
        residual = q
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        attn, attn_w = self.attention(q, k, v)
        attn = attn + residual
        return attn, attn_w


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.matmul(F.softmax(attn / k.shape[-1], dim=1), v)
        attn_w = F.softmax(attn, dim=1)
        return attn, attn_w


def main():
    pass


if __name__ == "__main__":
    main()
