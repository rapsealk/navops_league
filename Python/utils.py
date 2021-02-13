#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import socket
import traceback
from itertools import count

import numpy as np
import torch
from torch.autograd import Variable
from slack import WebClient
from slack.errors import SlackApiError


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


class SizeEstimator(object):
    """
    PyTorch Model Size Estimator (https://github.com/jacobkimmel/pytorch_modelsize)
    """
    def __init__(self, model, input_size=(1,1,32,32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []
        
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size))*self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total


@SlackNotification
def main():
    raise Exception("Hi")


if __name__ == "__main__":
    main()
