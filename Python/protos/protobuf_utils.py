#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

from learner_pb2 import Tensor, TensorArray


def encode_tensors(tensors):
    """
    Args:
        tensors: [np.array(dtype=np.float32), ...]
    """
    assert type(tensors) is list

    type_mapper = {
        np.dtype('float32'): 0,
        np.dtype('float64'): 1,
        np.dtype('int32'): 2,
        np.dtype('int64'): 3,
        np.dtype('uint32'): 4,
        np.dtype('uint64'): 5
    }

    tensors = [Tensor(data=tensor.tobytes(),
                      shape=tensor.shape,
                      dtype=type_mapper[tensor.dtype])
               for tensor in tensors]

    return TensorArray(data=tensors)


def decode_tensors(tensors):
    """
    Args:
        tensors: learner_pb2.TensorArray
    """
    assert type(tensors) is TensorArray

    type_mapper = [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.uint32,
        np.uint64
    ]

    tensors_np = [np.frombuffer(tensor.data, dtype=type_mapper[tensor.dtype]).reshape(tensor.shape)
                  for tensor in tensors.data]

    return tensors_np


if __name__ == "__main__":
    pass
