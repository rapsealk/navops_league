#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np

from board import ReportingBoard


def main():
    board = ReportingBoard(dirname=os.path.abspath(os.path.dirname(__file__)))
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

    board.plot_winning_rate(wins, draws, loses)

    n_step = 5
    mean_wins = [0] * (n_step - 1) + wins
    mean_draws = [0] * (n_step - 1) + draws
    # mean_loses = [0] * (n_step - 1) + loses

    new_wins = [int(np.mean(mean_wins[i:i+3])) for i in range(len(mean_wins)-n_step+1)]
    new_draws = [int(np.mean(mean_draws[i:i+3])) for i in range(len(mean_wins)-n_step+1)]
    new_loses = [100-new_wins[i]-new_draws[i] for i in range(len(mean_wins)-n_step+1)]

    board.plot_winning_rate(new_wins, new_draws, new_loses)


if __name__ == "__main__":
    main()
