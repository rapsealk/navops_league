#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime
from pprint import pprint
from uuid import uuid4

import numpy as np


class EloRatingManager:

    def __init__(self):
        self.p = 400

    def evaluate(self, r_a: int, r_b: int, a_wins: bool):
        q_a = pow(10, r_a / self.p)
        q_b = pow(10, r_b / self.p)
        prob_a = q_a / (q_a + q_b)
        prob_b = q_b / (q_a + q_b)
        b_wins = 1 - a_wins
        # change = round(K * (a_wins - prob_a))
        # r_a_ = r_a + change
        # r_b_ = r_b - change
        r_a_ = r_a + round(self.k(r_a) * (a_wins - prob_a))
        r_b_ = r_b + round(self.k(r_b) * (b_wins - prob_b))
        return r_a_, r_b_

    def k(self, rating: int):
        if rating > 2400:
            return 16
        elif rating > 2100:
            return 24
        return 32


class MatchHistory:
    """
    {
        "timestamp": "2021-01-27T10:56:31.345017",
        "players": ("aaaa", "bbbb"),
        "result": 0
    }
    """
    def __init__(self):
        self._history = []

    def add(self, history):
        assert type(history) is dict
        self._history.append(history)

    def winning_rate(self, id_):
        history = list(filter(lambda x: id_ in x["players"], self._history.copy()))
        victories = list(filter(lambda x: x["players"][x["result"]] == id_, history))
        return len(victories) / len(history)

    def winning_rate_vs(self, id1, id2):
        history = list(filter(lambda x: sorted((id1, id2)) == sorted(x["players"]), self._history.copy()))
        victories = list(filter(lambda x: x["players"][x["result"]] == id1, history))
        return len(victories) / len(history)


def generate_entities(n=16):
    return [
        {
            "id": str(uuid4()).replace('-', '')[:16],
            "rating": 1200
        }
        for _ in range(n)
    ]


def prioritized_fictitious_self_play(a, candidates):
    # f(P[A beats B]) / Sigma(f(P[A beats C]))
    # 1. n candidates
    # 2. sample with prioritized weight
    """
    def f(x):
        return x
    """
    winning_rates = [match_history.winning_rate_vs(a["id"], c["id"])
                     for c in candidates]
    priorities = np.array(winning_rates) / np.sum(winning_rates)
    candidate = np.random.choice(candidates, p=priorities)

    return candidate


if __name__ == "__main__":
    entities = generate_entities()

    # 1. Play
    rating = EloRatingManager()
    match_history = MatchHistory()

    for _ in range(4):
        for entity in entities:
            for opponent in entities:
                if entity is opponent:
                    continue
                result = np.random.choice([0, 1], p=np.array([entity["rating"], opponent["rating"]]) / (entity["rating"] + opponent["rating"]))
                match_history.add({
                    "timestamp": datetime.now().isoformat(),
                    "players": (entity["id"], opponent["id"]),
                    "result": result
                })

                entity["rating"], opponent["rating"] = rating.evaluate(entity["rating"], opponent["rating"], bool(1 - result))

    for entity in entities:
        entity["winning_rate"] = match_history.winning_rate(entity["id"])
    pprint(entities)

    opponent = prioritized_fictitious_self_play(entities[0], entities[1:])
    print('--------')
    pprint((sorted(entities, key=lambda x: -x["rating"])[0], sorted(entities, key=lambda x: -x["rating"])[-1]))
    pprint((sorted(entities, key=lambda x: -x["winning_rate"])[0], sorted(entities, key=lambda x: -x["winning_rate"])[-1]))
    pprint((entities[0], opponent, match_history.winning_rate_vs(entities[0]["id"], opponent["id"])))
