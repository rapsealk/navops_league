#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import abc
# import hashlib
import random
# from datetime import datetime
from uuid import uuid4
from collections import deque

"""
First phase (proposals)
(i) If x receives a proposal from y, then
    (a) he rejects it at once if he already holds a better proposal (i.e., a proposal from someone higher than y in his preference list);
    (b) he holds it for consideration otherwise, simultaneously rejecting any poorer proposal that he currently holds.
(ii) An individual x proposes to the others in the order in which they appear in his preference list, stopping when a promise of consideration is received;
     any subsequent rejectioncauses x to continue immediately his sequence of proposals.
"""


def stable_roommates_psuedo(_):
    """
    x = [
        {
            "id": str,
            "rating": int,
            "proposal": x,
            "holding": x,
            "prefs": [x]
        }
    ]
    """
    #assert len(x) % 2 == 0
    #q = deque(x)
    #while len(q) > 0:
    #    proposer = q.popleft()
    #    for pref in proposer["prefs"]:
    #        if pref[""]
    pass


class BasePlayer(abc.ABC):

    def __init__(self, name):
        self.name = name
        self.prefs = []
        self.matching = None

        self._pref_names = []
        self._original_prefs = None

    def __repr__(self):
        return str(self.name)

    def forget(self, other):
        prefs = self.prefs[:]
        prefs.remove(other)
        self.prefs = prefs

    def unmatched_message(self):
        return "{} is unmatched.".format(self)

    def not_in_preferences_message(self, other):
        return """
            {} is matched to {}, but they do not appear in their preference list: {}
        """.format(self, other, self.prefs)

    def set_prefs(self, players):
        self.prefs = players
        self._pref_names = [player.name for player in players]

    def prefers(self, player, other):
        """Determines whether the player prefers a player over some other player."""
        prefs = self._original_prefs
        return prefs.index(player) < prefs.index(other)


def roommates(entities):
    assert len(entities) % 2 == 0


class Roommate:

    def __init__(self):
        pass

    def propose(self):
        pass

    def proposed(self):
        pass

    def rejected(self):
        self.proposal = None


if __name__ == "__main__":
    """
    https://matching.readthedocs.io/en/latest/discussion/stable_roommates/
    https://github.com/daffidwilde/matching/blob/main/src/matching/base.py
    """
    entities = [
        {
            "id": str(uuid4()).replace('-', '')[:16],
            "rating": 1200 + random.randint(-400, 400),
            # "proposal": None,
            # "holding": None
        }
        for _ in range(16)
    ]

    # matching
    import matching
    from matching.games import StableMarriage

    sorted_entities = [entity["id"] for entity in sorted(entities, key=lambda x: x["rating"])]

    prefs = {
        entity["id"]: list(filter(lambda x: x != entity["id"], sorted_entities))
        # entity["id"]: list(map(lambda x: x["id"], sorted(list(filter(lambda x: x["id"] != entity["id"], entities)), key=lambda x: x["rating"])))
        for entity in entities
    }
    # print(prefs)

    game = StableMarriage.create_from_dictionaries()

    """
    for entity in entities:
        entity["candidates"] = [e for e in entities if e["id"] != entity["id"]]
        entity["candidates"] = sorted(entity["candidates"], key=lambda x: x["rating"])
    """
    # candidates = sorted(entities, lambda x: x["rating"])
    # assert len(candidates) % 2 == 0

    # proposer = candidates.pop(0)
    # while len(candidates) > 0:
    #     pass

    """
    # 1. Proposals
    proposal_queue = deque([e for e in entities])
    assert len(proposal_queue) % 2 == 0
    # criteria = lambda x, y: x["rating"] < y["rating"]

    while len(proposal_queue) > 0:
        entity = proposal_queue.popleft()
        print('entity:', entity["id"], len(entity["candidates"]), len(proposal_queue))
        assert entity["proposal"] is None
        # for partner in entities:
        for candidate in entity["candidates"].copy():
            entity["candidates"].remove(candidate)
            # if entity["id"] == partner["id"]:
            #     continue
            if candidate["holding"] is not None:
                if candidate["holding"]["rating"] > entity["rating"]:
                    candidate["holding"]["proposal"] = None
                    proposal_queue.insert(0, candidate["holding"])
                    candidate["holding"] = entity
                    entity["proposal"] = candidate
                    break
                else:   # rejected
                    continue    # proposal_queue.insert(0, entity)
            else:
                candidate["holding"] = entity
                entity["proposal"] = candidate
                break

    entity_ids = [e["id"] for e in entities]
    matches = []
    for entity in entities:
        if entity["id"] in entity_ids:
            continue
        entity_ids.remove(entity["id"])
        entity_ids.remove(entity["proposal"]["id"])
        matches.append((entity["id"], entity["proposal"]["id"]))

    for match in matches:
        print(match)
    print(len(matches), len(entity_ids))
    """
