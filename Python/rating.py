#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
P = 400
K = 32  # below 2100
K = 24  # 2100~2400 (by Jeff Sonas)
K = 16  # above 2400


class Rating:
    pass


class ELORating(Rating):

    @staticmethod
    def calc(r_a: int, r_b: int, a_wins: bool):
        q_a = pow(10, r_a / P)
        q_b = pow(10, r_b / P)
        prob_a = q_a / (q_a + q_b)
        # prob_b = q_b / (q_a + q_b)
        change = round(K * (a_wins - prob_a))
        r_a_ = r_a + change
        r_b_ = r_b - change
        return r_a_, r_b_


if __name__ == "__main__":
    pass
