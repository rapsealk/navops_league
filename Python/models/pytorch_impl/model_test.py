#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import os
import hashlib
from datetime import datetime

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, n_in, n_out):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_out)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class TestModelManagement(unittest.TestCase):

    def setUp(self):
        hashstring = hashlib.sha256(datetime.now().isoformat().encode("utf-8")).hexdigest()
        self.path = os.path.join(os.path.dirname(__file__), hashstring[:8] + '.ckpt')

    def test_model_load(self):
        n_in, n_out = (8, 1)
        parameters = torch.rand((4,))

        model1 = Model(n_in, n_out)
        torch.save({
            "state_dict": model1.state_dict(),
            "parameters": parameters
        }, self.path)

        model2 = Model(n_in, n_out)
        checkpoint = torch.load(self.path)
        model2.load_state_dict(checkpoint["state_dict"])

        inputs = torch.rand((n_in,))
        self.assertTrue(torch.all(model1(inputs) == model2(inputs)))
        self.assertTrue(torch.all(parameters == checkpoint["parameters"]))

    def tearDown(self):
        try:
            os.remove(self.path)
        except NotImplementedError:
            pass


if __name__ == "__main__":
    unittest.main()
