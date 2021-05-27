#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from collections import deque

import numpy as np

from plotboard import WinRateBoard


def main():
    board = WinRateBoard(dirpath=os.path.abspath(os.path.dirname(__file__)))
    """
    # x = ['100', '200', '300']
    x = np.linspace(1, 1000, 1000)
    wins = np.random.uniform(0.0, 0.4, (1000,))
    draws = np.random.uniform(0.0, 0.4, (1000,))
    loses = np.ones_like(wins) - wins - draws
    """
    wins = [
        21, 17, 14, 23, 14, 6, 10, 10, 37, 29,
        40, 22, 37, 37, 40, 48, 40, 34, 46, 43,
        39, 50, 41, 17, 29, 31, 34, 42, 51, 63,
        64, 46, 55, 77, 68, 58, 60, 58, 67, 71,
        69, 69
    ]
    draws = [
        0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 1, 1, 0, 0, 0, 1, 2,
        2, 2
    ]
    loses = [
        79, 82, 86, 77, 86, 94, 90, 89, 62, 71,
        60, 78, 62, 63, 60, 52, 59, 66, 54, 57,
        60, 49, 59, 83, 71, 69, 66, 57, 49, 37,
        36, 53, 45, 22, 31, 42, 40, 42, 32, 27,
        29, 29
    ]
    assert len(wins) == len(draws) and len(wins) == len(loses)

    x = [str(i*100) for i in range(1, 43)]
    """
    y = [
        [0.1, 0.2, 0.8],    # Win
        [0.2, 0.2, 0.1],    # Draw
        [0.7, 0.6, 0.1]     # Lose
    ]
    """
    y = [wins, draws, loses]
    board.plot(tuple(deque(x)), tuple(deque(y)), title=['Win', 'Draw', 'Lose'], title_texts=['Episode', 'Win Rates (per 100 eps)'])

    # board.plot_scatter(y)


if __name__ == "__main__":
    main()
