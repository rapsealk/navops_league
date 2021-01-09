#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import json

import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": [
            "192.168.4.121:55655",
            "192.168.4.123:55655"
        ]
    },
    "task": {
        "type": "worker",
        "index": 0
    }
})

NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS


def main():
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    with strategy.scope():
        pass


if __name__ == "__main__":
    main()
