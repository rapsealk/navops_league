#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest

from utils import to_radian, to_polar_coordinate


class CoordinateTest(unittest.TestCase):

    def test_polar_coordinates(self):
        theta = 0
        self.assertEqual(to_polar_coordinate(to_radian(theta)), (1, 0))
        # theta = 90
        # self.assertEqual(to_polar_coordinate(to_radian(theta)), (0, 1))
        # theta = 180
        # self.assertEqual(to_polar_coordinate(to_radian(theta)), (-1, 0))
        # theta = 270
        # self.assertEqual(to_polar_coordinate(to_radian(theta)), (0, -1))


if __name__ == "__main__":
    unittest.main()
