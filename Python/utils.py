#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from itertools import count


def epsilon(discount=1e-3, step=100, minimum=5e-3):
    value = 1.0
    for i in count(1):
        if i % step == 0:
            value = max(value - discount, minimum)
        yield value


if __name__ == "__main__":
    pass
