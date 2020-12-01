#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

from learner_pb2 import Tensor, Weights


def encode_weights(weights):
    """
    Args:
        weights: [np.array(dtype=np.float32), ...]
    """
    assert type(weights) is list

    type_mapper = {
        np.dtype('float32'): 0,
        np.dtype('float64'): 1,
        np.dtype('int32'): 2,
        np.dtype('int64'): 3,
        np.dtype('uint32'): 4,
        np.dtype('uint64'): 5
    }

    tensors = [Tensor(data=weight.tobytes(),
                      shape=weight.shape,
                      dtype=type_mapper[weight.dtype])
               for weight in weights]

    return Weights(data=tensors)


def decode_weights(weights):
    """
    Args:
        weights: learner_pb2.Weights
    """
    assert type(weights) is Weights

    type_mapper = [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.uint32,
        np.uint64
    ]

    weights_np = [np.frombuffer(tensor.data, dtype=type_mapper[tensor.dtype]).reshape(tensor.shape)
                  for tensor in weights.data]

    return weights_np


if __name__ == "__main__":
    pass
