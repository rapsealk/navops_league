#!/usr/bin/python3
# -*- coding: utf-8 -*-
from uuid import uuid4


def generate_id(k=16):
    return str(uuid4()).replace('-', '')[:k]


if __name__ == "__main__":
    pass
