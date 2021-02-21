#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import time
# from datetime import datetime

import grpc
from google.protobuf.empty_pb2 import Empty

try:
    from . import rimpac_pb2
    from . import rimpac_pb2_grpc
except:
    import rimpac_pb2
    import rimpac_pb2_grpc


class RimpacLeagueClient: # Grpc

    def __init__(self, addr='127.0.0.1', port=61084):
        channel = grpc.insecure_channel('{}:{}'.format(addr, port))
        self.stub = rimpac_pb2_grpc.RimpacLeagueServiceStub(channel)

    def request_match(self):
        print('RimpacLeagueClient: request_match')
        try:
            chunk = self.stub.Match(Empty())
            print('chunk:', chunk)
        except grpc._channel._InactiveRpcError as e:
            print('grpc._channel._InactiveRpcError:', e)
        except grpc._channel._MultiThreadedRendezvous as e:
            print('grpc._channel._MultiThreadedRendezvous:', e)
        except Exception as e:
            print('Exception:', e)

    def request_populate(self):
        print('RimpacLeagueClient: request_populate')
        request = rimpac_pb2.RequestPopulate(timestamp=time.time())
        try:
            response = self.stub.Populate(request)
        except grpc._channel._InactiveRpcError as e:
            print('grpc._channel._InactiveRpcError:', e)
        except grpc._channel._MultiThreadedRendezvous as e:
            print('grpc._channel._MultiThreadedRendezvous:', e)
        except Exception as e:
            print('Exception:', e)

        return response


if __name__ == "__main__":
    client = RimpacLeagueClient()
    client.request_match()
