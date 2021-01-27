#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import os
import hashlib
from datetime import datetime

import torch
import numpy as np
from gym.spaces import Box
# import torch.nn as nn

from sac import SoftActorCriticAgent


"""
class Model(nn.Module):

    def __init__(self, n_in, n_out):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_out)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
"""


class TestModelManagement(unittest.TestCase):

    def setUp(self):
        hashstring = hashlib.sha256(datetime.now().isoformat().encode("utf-8")).hexdigest()
        self.path = os.path.join(os.path.dirname(__file__), hashstring[:8] + '.ckpt')
        print('path:', self.path)

    def test_model_load(self):
        num_inputs = 8
        action_space = Box(-1.0, 1.0, shape=(2,))
        # n_in, n_out = (8, np.array([1]))
        # parameters = torch.rand((4,))

        model1 = SoftActorCriticAgent(num_inputs, action_space)
        model1.save(self.path)
        """
        model1 = Model(n_in, n_out)
        torch.save({
            "state_dict": model1.state_dict(),
            "parameters": parameters
        }, self.path)
        """

        model2 = SoftActorCriticAgent(num_inputs, action_space)
        model2.load(self.path)
        """
        model2 = Model(n_in, n_out)
        checkpoint = torch.load(self.path)
        model2.load_state_dict(checkpoint["state_dict"])
        """

        # inputs = torch.rand((1, 1, num_inputs))
        inputs = np.random.uniform(-1.0, 1.0, (1, num_inputs))
        inputs = torch.FloatTensor([inputs])
        print('inputs.shape:', inputs.shape)
        # f1 = model1.select_action(inputs)
        # f2 = model2.select_action(inputs)
        f1_mean, f1_log_std = model1.policy(inputs.to(model1.device))
        f2_mean, f2_log_std = model2.policy(inputs.to(model2.device))
        print('f1:', f1_mean)
        print('f2:', f2_mean)
        # self.assertTrue(np.all(f1 == f2))
        # self.assertTrue(np.all(model1.select_action(inputs) == model2.select_action(inputs)))
        # self.assertTrue(torch.all(parameters == checkpoint["parameters"]))
        self.assertTrue(np.all(f1_mean == f2_mean))

    def tearDown(self):
        try:
            os.remove(self.path)
        except NotImplementedError:
            pass
        # except FileNotFoundError:
        #     pass


if __name__ == "__main__":
    unittest.main()
