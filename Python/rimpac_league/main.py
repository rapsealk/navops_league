#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime
from uuid import uuid4

from services.impls.database import session
from services.impls.database.models import Model, MatchHistory

if __name__ == "__main__":
    # print(list(session.query(Model)))
    models = list(session.query(Model))

    id_ = str(uuid4()).replace('-', '')[:16]
    # path = os.path.abspath(__file__)
    model = Model(id=id_, rating=random.randint(1200, 1300), path='{}-{}'.format(id_, random.randint(0, 9) * 100))
    session.add(model)
    session.commit()

    match_history = MatchHistory(home=models[0].id, away=models[1].id, result=random.randint(0, 1), timestamp=datetime.now())
    print('MatchHistory.id:', match_history.id)
    print('MatchHistory.home:', match_history.home)
    print('MatchHistory.away:', match_history.away)
    print('MatchHistory.result:', match_history.result)
    print('MatchHistory.timestamp:', match_history.timestamp)
    session.add(match_history)
    session.commit()
