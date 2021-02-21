#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import grpc
from google.protobuf.empty_pb2 import Empty

import reward_pb2_grpc


class RibbonClient:

    def __init__(self, addr='127.0.0.1', port=61084):
        channel = grpc.insecure_channel('%s:%d' % (addr, port))
        self.stub = reward_pb2_grpc.RewardBridgeStub(channel)

    def get_ribbon(self):
        try:
            ribbon = self.stub.GetRibbon(Empty())
            return ribbon.id
        except grpc._channel._InactiveRpcError:
            pass
        return 0


if __name__ == "__main__":
    client = RibbonClient()
    print('GetRibbon:', client.get_ribbon())