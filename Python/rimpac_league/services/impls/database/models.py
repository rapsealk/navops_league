#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from uuid import uuid4

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Model(Base):

    __tablename__ = "model"

    id = Column(String(20), primary_key=True)
    rating = Column(Integer)
    path = Column(String(100))

    """
    def __init__(self, id: str, rating: int, path: str):
        self.id = id    # str(uuid4()).replace('-', '')[:16]
        self._rating = rating
        self._path = path
    """

    """
    @property
    def id(self):
        return self._id
    """

    """
    @property
    def rating(self):
        return self._rating

    @property
    def path(self):
        return self._path
    """

    def __repr__(self):
        return "<Model({}, {}, {})>".format(self.id, self.rating, self.path)


class MatchHistory(Base):

    __tablename__ = "match_history"

    id = Column(Integer, primary_key=True)
    home = Column(String(20), ForeignKey('model.id'))
    away = Column(String(20), ForeignKey('model.id'))
    result = Column(Integer)
    timestamp = Column(DateTime)

    def __repr__(self):
        return "<MatchHistory({}, {}, {}, {}, {})>".format(self.id, self.player1, self.player2, self.result, self.timestamp)
