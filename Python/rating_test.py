#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest

from rating import ELORating


class ELORatingTestCase(unittest.TestCase):

    def test_new_rating(self):
        alpha = 2600
        beta = 2300
        new_alpha, new_beta = ELORating.calc(alpha, beta, True)
        self.assertEqual(new_alpha, 2602)
        self.assertEqual(new_beta, 2298)
        new_alpha, new_beta = ELORating.calc(alpha, beta, False)
        self.assertEqual(new_alpha, 2586)
        self.assertEqual(new_beta, 2314)


if __name__ == "__main__":
    unittest.main()
