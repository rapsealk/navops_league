#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import deque

import numpy as np

from plotboard import WinRateBoard


def main():
    board = WinRateBoard()
    # x = ['100', '200', '300']
    x = np.linspace(1, 1000, 1000)
    wins = np.random.uniform(0.0, 0.4, (1000,))
    draws = np.random.uniform(0.0, 0.4, (1000,))
    loses = np.ones_like(wins) - wins - draws
    """
    y = [
        [0.1, 0.2, 0.8],    # Win
        [0.2, 0.2, 0.1],    # Draw
        [0.7, 0.6, 0.1]     # Lose
    ]
    """
    y = [wins, draws, loses]
    board.plot(tuple(deque(x)), tuple(deque(y)))

    board.plot_scatter(y)


if __name__ == "__main__":
    main()
