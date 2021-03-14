#/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import json
from concurrent import futures
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
import pymongo
import grpc
from google.protobuf import empty_pb2

import league_pb2
import league_pb2_grpc

with open(os.path.join(os.path.dirname(__file__), '..', 'python', 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    MONGO_HOSTNAME = config["mongo"]["hostname"]
    MONGO_USERNAME = config["mongo"]["username"]
    MONGO_PASSWORD = config["mongo"]["password"]

DEFAULT_RATING = 1200
GRPC_CHUNK_SIZE = 1024 * 64

parser = argparse.ArgumentParser()
parser.add_argument('--client', action='store_true', default=False)
args = parser.parse_args()


def generate_id():
    return str(uuid4()).replace('-', '')


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 64)
        self.linear_h = nn.Sequential(
            nn.Linear(64, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 64)
        )
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.linaer_h(x))
        return self.value(x)


class MongoDatabase:

    def __init__(
        self,
        username=MONGO_USERNAME,
        password=MONGO_PASSWORD,
        hostname=MONGO_HOSTNAME
    ):
        client = pymongo.MongoClient(f'mongodb://{username}:{password}@{hostname}:27017/')
        self._database = client["rimpac"]
        # self._collection = database["cgf"]

    def create(self, table: str, data: dict):
        collection = self._database[table]
        return collection.insert_one(data)

    def read(self, table: str, key: str):
        collection = self._database[table]
        return collection.find_one({"_id": key})

    def update(self, table: str, key: str, data: dict):
        collection = self._database[table]
        return collection.update_one(
            {"_id": key},
            {"$set": {**data}}
        )

    def delete(self, table: str, key: str):
        collection = self._database[table]
        return collection.delete_one({"_id": key})

    def sample(self, table: str, n: int):
        collection = self._database[table]
        pipeline = [
            {"$sample": {"size": n}}
        ]
        return collection.aggregate(pipeline)


class LeagueManagementServer(league_pb2_grpc.LeagueManagementServicer):

    def __init__(self, database=None):
        self._database = database or MongoDatabase()

    def RegisterCgf(self, request, context):
        """
        Args:
            request: RegisterCgfRequest {
                cgf: Cgf {
                    id: string,
                    path: string,
                    rating: uint32
                }
            }
        Returns:
            response: RegisterCgfResponse {
                code: StatusCode
            }
        """
        result = self._database.create(
            table='cgf',
            data={
                "path": request.cgf.path,
                "rating": request.cgf.rating
            }
        )
        print(f'[server] register cgf: {result.inserted_id} ({type(result.inserted_id)})')
        return league_pb2.RegisterCgfResponse(
            code=league_pb2.StatusCode.Ok,
            id=str(result.inserted_id)
        )

    def GetRandomCgf(self, request, context):
        """
        Args:
            request: google.protobuf.empty_pb2.Empty
        Returns:
            response: GetRandomCgfResponse {
                code: StatusCode,
                cgf: Cgf
            }
        """
        cgf = self._database.sample(table='cgf', n=1)
        cgf = list(cgf)
        if len(cgf) == 0:
            return league_pb2.GetRandomCgfResponse(
                code=league_pb2.StatusCode.Failed
            )
        cgf = cgf[0]
        print('cgf:', cgf)
        cgf = league_pb2.Cgf(id=str(cgf["_id"]), path=cgf["path"], rating=cgf["rating"])
        return league_pb2.GetRandomCgfResponse(
            code=league_pb2.StatusCode.Ok,
            cgf=cgf
        )

    """
    def GetCgfInfo(self, request, context):
        # Args:
        #     request: GetCgfInfoRequest {
        #         id: string
        #     }
        # Returns:
        #     response: GetCgfInfoResponse {
        #         code: StatusCode,
        #         cgf: Cgf
        #     }
        result = self._database.read(table='cgf', key=request.id)
        print('[server] GetCgfInfo:', result)
        if result is None:
            return league_pb2.GetCgfInfoResponse(
                code=league_pb2.StatusCode.Failed
            )
        return league_pb2.GetCgfInfoResponse(
            code=league_pb2.StatusCode.Ok,
            cgf=league_pb2.Cgf(
                id=result["id"],
                path=result["path"],
                rating=result["rating"]
            )
        )

    def UpdateCgfInfo(self, request, context):
        # Args:
        #     request: UpdateCgfInfoRequest {
        #         id: string,
        #         path: string?,
        #         rating: uint32?
        #     }
        # Returns:
        #     response: UpdateCgfInfoResponse {
        #         code: StatusCode
        #     }
        new_data = {}
        if not not request.path:
            new_data["path"] = request.path
        if not not request.rating:
            new_data["rating"] = request.rating
        _ = self._database.update(table='cgf', key=request.id, data=new_data)
        return league_pb2.UpdateCgfInfoResponse(code=league_pb2.StatusCode.Ok)

    def DeleteCgfInfo(self, request, context):
        # Args:
        #     request: DeleteCgfInfoRequest {
        #         id: string
        #     }
        # Returns:
        #     response: DeleteCgfInfoResponse {
        #         code: StatusCode
        #     }
        _ = self._database.delete(table='cgf', key=request.id)
        return league_pb2.DeleteCgfInfoResponse(code=league_pb2.StatusCode.Ok)
    """

    def UploadCheckpoint(self, request_iterator, context):
        """
        Args:
            request_iterator: [UploadCheckpointRequest] {
                chunk: Chunk {
                    data: bytes
                }
            }
        Returns:
            response: UploadCheckpointResponse {
                code: StatusCode,
                path: string
            }
        """
        bytes_ = b''
        for request in request_iterator:
            bytes_ += request.chunk.data
            print(f'Server: {len(request.chunk.data)} ({context})')
        path = os.path.join(os.path.dirname(__file__), generate_id() + '.pth.tar')
        with open(path, 'wb') as f:
            f.write(bytes_)
        return league_pb2.UploadCheckpointResponse(
            code=league_pb2.StatusCode.Ok,
            path=path
        )

    def DownloadCheckpoint(self, request, context):
        bytes_ = b''
        with open(os.path.join(os.path.dirname(__file__), request.path), 'rb') as f:
            bytes_ = f.read()
        code = league_pb2.StatusCode.Ok
        while bytes_:
            try:
                yield league_pb2.DownloadCheckpointResponse(
                    code=code,
                    chunk=league_pb2.Chunk(data=bytes_[:GRPC_CHUNK_SIZE])
                )
                bytes_ = bytes_[GRPC_CHUNK_SIZE:]
            except:
                code = league_pb2.StatusCode.Failed
                yield league_pb2.DownloadCheckpointResponse(
                    code=code
                )
                break
        print(f'[Server] DownloadCheckpoint job is done! ({code})')


class LeagueManagementClient:

    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self._stub = league_pb2_grpc.LeagueManagementStub(channel)

    def register_cgf(self, path: str):
        cgf = league_pb2.Cgf(id=None, rating=DEFAULT_RATING, path=path)
        request = league_pb2.RegisterCgfRequest(cgf=cgf)
        response = self._stub.RegisterCgf(request)
        # if response.code == league_pb2.StatusCode.Ok
        return response.id

    def get_random_cgf(self):
        request = empty_pb2.Empty()
        response = self._stub.GetRandomCgf(request)
        print(f'get_random_cgf: {response.code == league_pb2.StatusCode.Ok}')
        if response.code != league_pb2.StatusCode.Ok:
            return None
        return response.cgf

    """
    def get_cgf(self, id_: str):
        request = league_pb2.GetCgfInfoRequest(id=id_)
        response = self._stub.GetCgfInfo(request)
        if response.code != league_pb2.StatusCode.Ok:
            return None
        return response.cgf

    def update_cgf(self, id_: str, path: str = None, rating: int = None):
        request = league_pb2.UpdateCgfInfoRequest(id=id_, path=path, rating=rating)
        response = self._stub.UpdateCgfInfo(request)
        return response.code

    def delete_cgf(self, id_: str):
        request = league_pb2.DeleteCgfInfoRequest(id=id_)
        response = self._stub.DeleteCgfInfo(request)
        return response.code
    """

    def upload_checkpoint(self, path: str):
        request_iterator = self._upload_request_iterator(path)
        response = self._stub.UploadCheckpoint(request_iterator)
        return response.code, response.path

    def download_checkpoint(self, path: str):
        request = league_pb2.DownloadCheckpointRequest(path=path)
        future_response = self._stub.DownloadCheckpoint(request)
        bytes_ = b''
        for response in future_response:
            if response.code != league_pb2.StatusCode.Ok:
                return None
                break
            bytes_ += response.chunk.data
        saved_path = os.path.join(os.path.dirname(__file__), generate_id() + '.pth.tar')
        with open(saved_path, 'wb') as f:
            f.write(bytes_)
        return saved_path

    def _upload_request_iterator(self, path: str):
        bytes_ = b''
        with open(path, 'rb') as f:
            bytes_ = f.read()
        total_bytes = len(bytes_)
        bytes_sent = 0
        while bytes_:
            chunk = bytes_[:GRPC_CHUNK_SIZE]
            bytes_ = bytes_[GRPC_CHUNK_SIZE:]
            bytes_sent += len(chunk)
            print(f'Client: {bytes_sent} / {total_bytes} ({bytes_sent / total_bytes * 100}%)')
            yield league_pb2.UploadCheckpointRequest(
                chunk=league_pb2.Chunk(data=chunk)
            )


def main():
    if args.client:
        client = LeagueManagementClient()
        path = os.path.join(os.path.dirname(__file__), 'checkpoint.pth.tar')
        model = Model()
        torch.save({"state_dict": model.state_dict()}, path)
        _id = client.register_cgf(path)
        print(f'_id: {_id}')
        cgf = client.get_random_cgf()
        print(f'cgf: {cgf}')
        return

        # code, saved_path = client.upload_checkpoint(path)
        # print('Client: code:', code == league_pb2.StatusCode.Ok)
        # checkpoint = torch.load(saved_path)
        # model.load_state_dict(checkpoint["state_dict"])
        # os.remove(saved_path)   # ...
        """
        response = client.register_cgf(
            id=str(uuid4()).replace('-', ''),
            path=path
        )
        print(f'[client] register cgf: {response}')
        """
        local_path = client.download_checkpoint(path)
        if local_path is not None:
            checkpoint = torch.load(local_path)
            model.load_state_dict(checkpoint["state_dict"])
    else:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        league_pb2_grpc.add_LeagueManagementServicer_to_server(LeagueManagementServer(), server)
        server.add_insecure_port('[::]:50051')
        print('Server is running..')
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    main()
