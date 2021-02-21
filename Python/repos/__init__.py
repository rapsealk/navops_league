#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
import abc

import sqlalchemy
# from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session, sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from database import Base


class Model(abc.ABC):
    pass


class Captain(Model):
    def __init__(self):
        self._id = -1

    @property
    def id(self):
        return self._id


class MatchHistory(Model):
    def __init__(self):
        self._captains = (0, 1)


class CaptainRepository:
    def __init__(self):
        engine = sqlalchemy.create_engine('mysql+pymysql://root:1111@localhost/rimpac')
        # session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        self._conn = engine.connect()
        metadata = sqlalchemy.MetaData()
        self._table = sqlalchemy.Table('test', metadata, autoload=True, autoload_with=engine)
        # print('table:', self._table.columns.keys())

    def get(self, uid):
        query = sqlalchemy.select([self._table])


class MatchHistoryRepository:
    pass


if __name__ == "__main__":
    pass
"""
