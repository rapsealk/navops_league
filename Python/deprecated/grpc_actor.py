#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import argparse

import grpc

import learner_pb2
import learner_pb2_grpc

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', type=str, default='localhost')
args = parser.parse_args()


if __name__ == "__main__":
    channel = grpc.insecure_channel('%s:61084' % args.server)
    stub = learner_pb2_grpc.LearnerServiceStub(channel)

    episode = learner_pb2.Episode()
    stub.Enqueue(episode)
