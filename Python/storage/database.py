#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sqlalchemy as db


def main():
    engine = db.create_engine('mysql+pymysql://root:1111@localhost/rimpac')
    connection = engine.connect()
    metadata = db.MetaData()
    table = db.Table('test', metadata, autoload=True, autoload_with=engine)
    print('table:', table.columns.keys())


if __name__ == "__main__":
    main()
