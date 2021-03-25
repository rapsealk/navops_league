#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import os
import json

import gym
import gym_navops   # noqa: F401

with open(os.path.join(os.path.dirname(__file__), 'gym-navops', 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)


class TestNavOpsEnvironment(unittest.TestCase):

    def setUp(self):
        self._env = gym.make('NavOps-v0', no_graphics=True)
        self._env.reset()

    def test_environment_space(self):
        self.assertEqual(self._env.observation_space.shape, tuple(config["observation_space"]["shape"]))
        self.assertEqual(self._env.action_space.shape, tuple(config["action_space"]["shape"]))

    def tearDown(self):
        self._env.close()


if __name__ == "__main__":
    unittest.main()
