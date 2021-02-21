#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import concurrent
import time
from datetime import datetime
from uuid import uuid4

import grpc
from google.protobuf.empty_pb2 import Empty

import rimpac_pb2
import rimpac_pb2_grpc


class RimpacLeagueServer(rimpac_pb2_grpc.RimpacLeagueServiceServicer):

    def __init__(self):
        pass

    def Match(self, request, context):
        print('[{}] Match'.format(datetime.now().isoformat()))
        return Empty()

    def Populate(self, request, context):
        print('[{}] Populate'.format(datetime.now().isoformat()))
        response = rimpac_pb2.ResponsePopulate(**{
            "timestamp": time.time(),
            "id": str(uuid4()).replace('-', '')[:16]
        })
        return response


if __name__ == "__main__":
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=8))
    rimpac_pb2_grpc.add_RimpacLeagueServiceServicer_to_server(RimpacLeagueServer(), server)
    server.add_insecure_port('[::]:61084')
    server.start()
    server.wait_for_termination()
