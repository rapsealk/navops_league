#!/bin/bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. $1
# python -m grpc_tools.protoc -I. --python_out=../python --grpc_python_out=../python $1