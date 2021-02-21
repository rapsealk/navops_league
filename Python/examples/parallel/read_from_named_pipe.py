#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

named_pipe = "my_pipe"

if not os.path.exists(named_pipe):
    os.mkfifo(named_pipe)


def read_message(input_pipe):
    # fd = os.open(input_pipe, os.O_RDONLY)
    with os.open(input_pipe, os.O_RDONLY) as fd:
        message = "I pid [%d] received a message => %s" % (os.getpid(), os.read(fd, 22))
    # os.close(fd)
    return message


def write_message(input_pipe, message):
    # fd = os.open(input_pipe, os.O_WRONLY)
    with os.open(input_pipe, os.O_WRONLY) as fd:
        os.write(fd, message % str(os.getpid()))
    # os.close(fd)


if __name__ == "__main__":
    pass
