#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def main():
    np.random.seed(4)
    print(np.random.rand(4))
    print(np.random.rand(4))
    np.random.seed(4)
    print(np.random.rand(4))


if __name__ == "__main__":
    main()
