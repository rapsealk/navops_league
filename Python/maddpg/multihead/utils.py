#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_to_tensor(device, *args):
    # return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))
    # return map(lambda tensor: tensor.float().to(device), map(torch.tensor, args))
    return torch.stack(tuple(map(lambda tensor: tensor.float().to(device), map(torch.from_numpy,
        map(lambda arr: np.array(arr, dtype=np.float32), args)
    ))))


def onehot(items, max_range=10):
    x = np.zeros((items.shape[0], max_range))
    x[range(items.shape[0]), items] = 1
    return torch.from_numpy(x).view(-1, max_range)


def weights_init_(m, activation_fn=F.relu):
    """
    TODO: Closure
    def weights_init_fn_(m):
        if activation_fn is ...:
            ...
    return weights_init_fn_
    """
    if isinstance(m, nn.Linear):
        # ReLU
        if activation_fn is F.relu:
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)
        # SiLU
        elif activation_fn is F.silu:
            torch.nn.init.xavier_normal_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    pass
