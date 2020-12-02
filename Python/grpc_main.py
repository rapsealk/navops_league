#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import concurrent
import queue
from collections import deque
from datetime import datetime
from threading import Thread
from multiprocessing import Queue, cpu_count

import tensorflow as tf
import numpy as np
import grpc
from google.protobuf.empty_pb2 import Empty

import learner_pb2
import learner_pb2_grpc

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

TENSORBOARD_WRITER = None
RESULTS = deque([0] * 100, maxlen=100)


class LearnerServer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(self):
        self.model = None
        self.queue = Queue()

        self.training_thread = Thread(daemon=True)
        self.training_thread.isDaemon = True
        self.training_thread.start()

    def Enqueue(self, request, context):
        assert type(request) is learner_pb2.Trajectory

        print('Learner.Enqueue:', len(request.observations), request.result)
        self.queue.put(request)

        return Empty()

    def train(self):
        print('Learner.train')
        while True:
            try:
                episode = self.queue.get()
            except queue.Empty as e:
                sys.stderr.write('[queue.Empty] %s\n' % e)
                continue

            print('[%s] Learner.train' % datetime.now().isoformat())

            with TENSORBOARD_WRITER.as_default():
                tf.summary.scalar('Reward', np.sum(returns), CURRENT_EPISODE)
                tf.summary.scalar('Loss', loss, CURRENT_EPISODE)
                tf.summary.scalar('Rate', np.mean(RESULTS), CURRENT_EPISODE)

            # observations = np.array(trajectory.observations, dtype=np.float32)
            # actions = np.array()


def main():
    learner = LearnerServer()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()))
    learner_pb2_grpc.add_LearnerServiceServicer_to_server(learner, server)
    server.add_insecure_port('[::]:61084')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    global TENSORBOARD_WRITER

    TENSORBOARD_WRITER = tf.summary.create_file_writer(os.path.join(os.path.dirname(__file__), 'summary', 'distributed', str(int(time.time() * 1000))))

    main()
