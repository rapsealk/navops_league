#!/usr/bin/python3
# -*- coding: utf-8 -*-
from uuid import uuid4

import firebase_admin


class Unit:
    def __init__(self):
        self.id = str(uuid4()).replace('-', '')     # len: 32
        self.path = ''
        self.rating = 1200


class ModelInfo:
    def __init__(self):
        """
            "type": type(model),
            "state_dict": model.state_dict(),
            "opt_type": type(opt),
            "opt_state_dict": opt.state_dict()
        """
        pass


class League:

    def __init__(self):
        self._database = None
        self._match_history = None
        self._pool = {}

    def request_model(self):
        pass


if __name__ == "__main__":
    pass
