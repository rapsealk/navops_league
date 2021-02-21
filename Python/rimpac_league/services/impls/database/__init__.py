# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, Table, Column, Integer, String
from sqlalchemy.orm import scoped_session, sessionmaker
# from sqlalchemy.ext.declarative import declarative_base

from .models import Base


engine = create_engine('mysql+pymysql://root:1111@localhost/rimpac')
session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base.query = session.query_property()
Base.metadata.create_all(engine)

"""
models = Table('model', Base.metadata,
               Column('id', String(20), primary_key=True),
               Column('rating', Integer),
               Column('path', String(100)))
"""
