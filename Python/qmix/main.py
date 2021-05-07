#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse

from models import QMix

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.98)
args = parser.parse_args()


def main():
    model = QMix()


if __name__ == "__main__":
    main()
