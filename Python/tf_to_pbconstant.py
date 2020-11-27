#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

if tf.__version__.startswith('2'):
    TF_VERSION = 2
    from tensorflow.python.framework import convert_to_constants
elif tf.__version__.startswith('1'):
    TF_VERSION = 1


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()

        self.dense = tf.keras.layers.Dense(8, activation='relu', input_shape=(4,))
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense(inputs)
        return self.out(x)


model = Model()


@tf.function(input_signature=[tf.TensorSpec(shape=[4, 1], dtype=tf.float32)])
def to_save(x):
    return model(x)


if __name__ == "__main__":
    print(tf.__version__)

    f = to_save.get_concrete_function()
    constant_graph = convert_to_constants.convert_variables_to_constants_v2(f)
    tf.io.write_graph(constant_graph.graph.as_graph_def(), 'pretrained', 'model.pb')
