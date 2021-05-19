#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import socket
import traceback
from itertools import count
from threading import Lock
from uuid import uuid4

import numpy as np
from slack import WebClient
from slack.errors import SlackApiError


def generate_id(k=16):
    return str(uuid4()).replace('-', '')[:k]


class Atomic:

    def __init__(self, dtype=int):
        self._value = dtype()
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            value = self._value
        return value


def epsilon(discount=1e-3, step=100, minimum=5e-3, initial_value=1.0):
    value = initial_value
    for i in count(1):
        if i % step == 0:
            value = max(value - discount, minimum)
        yield value


def discount_rewards(rewards: np.ndarray, gamma=0.998):  # 347 -> 0.5
    discounted = [0] * len(rewards)
    # discounted = np.zeros_like(rewards)
    discounted[-1] = rewards[-1]
    # print('before:', discounted)
    n = len(rewards) - 1
    for i in range(n):
        # print(f'[{n-i-1}] <- [{n-i}]({discounted[n-i]}) * gamma + [{n-i-1}]({rewards[n-i-1]})')
        x = discounted[n-i] * gamma
        # print('x1:', x)
        x = x + rewards[n-i-1]
        # print('x2:', x)
        discounted[n-i-1] = x
        # discounted[n-i-1] = discounted[n-i] * gamma + rewards[n-i-1]
        # print(rewards, discounted, discounted[n-i-1], n-i-1, gamma)
    # print('after:', discounted)
    return np.array(discounted)


"""
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
"""


class SlackNotification:
    def __init__(self, token: str):
        self._token = token

    def __call__(self, function):
        def decorator(*args, **kwargs):
            try:
                function(*args, **kwargs)
            except Exception as e:
                self._client = WebClient(token=self._token)
                message = f'[{socket.gethostname()}]\n{e}\n{traceback.format_exc()}'
                self._send_slack_message(message)
        return decorator

    def _send_slack_message(self, message):
        try:
            response = self._client.chat_postMessage(
                channel='#notification',
                text=message
            )
            sys.stdout.write(f'[SlackApi] response: {response}')
        except SlackApiError as e:
            assert e.response["ok"] is False
            assert e.response["error"]
            sys.stderr.write(f'[SlackApiError] {e.response["error"]}')


def main():
    pass


if __name__ == "__main__":
    main()
