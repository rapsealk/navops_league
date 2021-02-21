#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from rdsf.client import RimpacLeagueClient


if __name__ == "__main__":
    client = RimpacLeagueClient()
    response = client.request_populate()
    print('response:', response.id)
