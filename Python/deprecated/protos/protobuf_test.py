#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from protobuf_utils import encode_tensors, decode_tensors

tf.config.experimental.set_visible_devices([], 'GPU')


class ProtobufTestCase(unittest.TestCase):

    def setUp(self):
        self.input_shape = (4,)
        self.model = tf.keras.Sequential([
            tf.keras.Input((4,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

    def test_weights_tensor(self):
        inputs = np.random.uniform(-1, 1, (1, *self.input_shape))
        output = self.model(inputs)

        weights = self.model.get_weights()
        weights_proto = encode_tensors(weights)
        new_weights = decode_tensors(weights_proto)
        self.assertTrue(all([np.all(x == y)
                             for x, y in zip(weights, new_weights)]))

        self.model.set_weights(new_weights)
        new_output = self.model(inputs)
        self.assertTrue((output.numpy() == new_output.numpy()).all())


if __name__ == "__main__":
    unittest.main()
