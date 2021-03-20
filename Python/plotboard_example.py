#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from collections import deque

from plotboard import WinRateBoard


def main():
    board = WinRateBoard()
    x = ['100', '200', '300']
    y = [
        [0.1, 0.2, 0.8],    # Win
        [0.2, 0.2, 0.1],    # Draw
        [0.7, 0.6, 0.1]     # Lose
    ]
    board.plot(tuple(deque(x)), tuple(deque(y)))
    # board.plot(deque(x).list(), deque(y).list())

    x = ['100']
    y = [
        [0.1],    # Win
        [0.2],    # Draw
        [0.7]     # Lose
    ]
    board.plot(x, y)


if __name__ == "__main__":
    main()
