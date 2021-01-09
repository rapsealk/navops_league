#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def to_radian(degree):
    return degree / 180 * np.pi


def to_degree(radian):
    return radian / np.pi * 180


def to_polar_coordinate(theta):
    return np.cos(theta), np.sin(theta)


if __name__ == "__main__":
    pass
