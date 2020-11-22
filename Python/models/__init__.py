#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        pass


if __name__ == "__main__":
    writer = tf.summary.create_file_writer(os.path.join(os.path.dirname(__file__), 'summary'))
    # writer = tf.summary.SummaryWriter()
    for i in range(100):
        with writer.as_default():
            tf.summary.scalar('Episode', np.random.uniform(0, 100), i)
            tf.summary.scalar('Episode_', np.random.randint(100, 1000), i)
