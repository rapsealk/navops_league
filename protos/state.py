#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

from state_pb2 import Batch
from tensor_pb2 import Tensor


def main():
    observation = np.random.uniform(-1.0, 1.0, (64, 75))
    action = np.random.randint(10, size=(1,))
    next_observation = np.random.uniform(-1.0, 1.0, (64, 75))
    probability = np.random.uniform(0.0, 1.0, size=(1,))
    hidden_in = torch.zeros((1, 1, 1, 256)).float()
    hidden_out = torch.ones((1, 1, 1, 256)).float()

    hidden_in_np = hidden_in.numpy().astype(np.float64)
    hidden_out_np = hidden_out.numpy().astype(np.float64)

    # np.frombuffer
    batch = Batch(
        observation=Tensor(data=observation.tobytes(), shape=observation.shape, dtype=Tensor.Type.FLOAT32),
        action=action.tolist(),
        reward=np.random.normal(0, 1),
        next_observation=Tensor(data=next_observation.tobytes(), shape=next_observation.shape, dtype=Tensor.Type.FLOAT32),
        probability=probability.tolist(),
        hidden_in=Tensor(data=hidden_in_np.tobytes(), shape=hidden_in_np.shape, dtype=Tensor.Type.FLOAT32),
        hidden_out=Tensor(data=hidden_out_np.tobytes(), shape=hidden_out_np.shape, dtype=Tensor.Type.FLOAT32),
        done=False
    )
    # print(batch)

    observation_ = np.frombuffer(batch.observation.data).reshape(batch.observation.shape)   # .astype(batch.observation.dtype)
    print(np.all(observation == observation_))

    hidden_in_ = np.frombuffer(batch.hidden_in.data).reshape(batch.hidden_in.shape)
    hidden_out_ = np.frombuffer(batch.hidden_out.data).reshape(batch.hidden_out.shape)
    hidden_in_ = torch.tensor(hidden_in_.copy()).float()
    hidden_out_ = torch.tensor(hidden_out_.copy()).float()
    # print(hidden_in_, hidden_in_.shape, hidden_in_.dtype)
    # print(hidden_out_, hidden_out_.shape, hidden_out_.dtype)
    print(all((hidden_in == hidden_in_).squeeze()))


if __name__ == "__main__":
    main()
