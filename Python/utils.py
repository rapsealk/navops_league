#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import traceback
from itertools import count


def epsilon(discount=1e-3, step=100, minimum=5e-3):
    value = 1.0
    for i in count(1):
        if i % step == 0:
            value = max(value - discount, minimum)
        yield value


class LogErrorTrace:
    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        try:
            self._function(*args, **kwargs)
        except Exception as e:
            dirname = os.path.join(os.path.dirname(__file__), 'logs')
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            with open(os.path.join(dirname, f'{int(time.time() * 1000)}.log'), 'w') as f:
                f.write(str(e) + '\n' + traceback.format_exc())


@LogErrorTrace
def main():
    print('11')
    raise Exception("Hi")


if __name__ == "__main__":
    main()
