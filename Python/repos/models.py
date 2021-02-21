#!usr/local/bin/python3
# -*- coding: utf-8 -*-
from sqlalchemy import Column, Integer, String, DateTime
# from database import Base
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Captain(Base):
    """
    https://docs.sqlalchemy.org/en/13/orm/tutorial.html
    """
    __tablename__ = "captain"

    id = Column(Integer, primary_key=True)
    rating = Column(Integer)
    recent_10_games = Column(Integer)

    def __init__(self, rating):
        self._rating = rating

    @property
    def rating(self):
        return self._rating

    def __repr__(self):
        return "<Captain({}, {})>".format(self.id, self._rating)


class MatchHistory(Base):
    __tablename__ = "match_history"


class TableTest(Base):
    __tablename__ = 'test'

    id = Column(Integer, primary_key=True)
    #datatime = Column(DateTime)
    #string = Column(String(250))
    rating = Column(Integer)

    def __init__(self, rating):
        self.rating = rating

    def __repr__(self):
        return "<Test({}, {})>".format(self.id, self.rating)
