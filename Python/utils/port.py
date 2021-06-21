#!/usr/bin/python3
# -*- coding: utf-8 -*-
import socket
from contextlib import closing


def get_free_port(base_port=9090):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        for port in range(base_port, 1024*64):
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
        return -1


if __name__ == "__main__":
    port = get_free_port()
    print('port:', port)
