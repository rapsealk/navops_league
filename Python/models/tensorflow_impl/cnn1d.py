#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Cnn1dModel(tf.keras.Model):

    def __init__(self, input_size):
        super(Cnn1dModel, self).__init__()

        # self.cnn = tf.keras.layers.Conv1D(filters=32, kernel_size=input_size[1], strides=1, padding='valid')
        self.cnn = tf.keras.layers.Conv1D(filters=32, kernel_size=2, strides=1, padding='valid')

    def call(self, inputs):
        y = self.cnn(inputs)
        return y


def main():
    batch_size = 1
    time_horizon = 4
    observation_size = 8
    # input_size = (4, 8)
    input_size = (batch_size, time_horizon, observation_size)

    inputs = np.random.uniform(-1.0, 1.0, input_size)
    print(inputs, inputs.shape)
    # inputs = np.swapaxes(inputs, axis1=1, axis2=2)
    # print(inputs, inputs.shape)

    model = Cnn1dModel(input_size)
    y = model(inputs)
    print(y, y.shape)

    """
    return
    # input_size = (64, 1, 70)
    # inputs = np.random.uniform(-1.0, 1.0, input_size)
    # inputs = np.transpose(inputs[: 1:])
    inputs = np.swapaxes(inputs, axis1=1, axis2=2)
    print(inputs.shape)
    y = model(inputs)
    print(y.shape)
    """

    print(tf.keras.layers.Flatten()(y).shape)


if __name__ == "__main__":
    main()
