#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from database import init
from database import session
from models import TableTest


def show_tables():
    queries = session.query(TableTest)
    entries = [{"id": q.id, "rating": q.rating} for q in queries]
    print(entries)
    return entries


def add_entry(rating):
    table = TableTest(rating)
    session.add(table)
    session.commit()


def update_entry(id):
    session.query(TableTest).filter().update


def delete_entry(id):
    session.query(TableTest).filter(TableTest.id==id).delete()
    session.commit()


def main():
    init()
    add_entry(1200)
    entries = show_tables()
    delete_entry(entries[0]["id"])
    session.close()


if __name__ == "__main__":
    main()
