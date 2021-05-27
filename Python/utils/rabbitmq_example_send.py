#!/usr/bin/python3
# -*- coding: utf-8 -*-
import time
import json
from datetime import datetime

import pika


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='192.168.35.126'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')

    while True:
        message = input()
        if message == 'q':
            break

        t = time.thread_time()
        message = {"timestamp": time.time(), "message": message}
        message = json.dumps(message)
        channel.basic_publish(exchange='',
                              routing_key='hello',
                              body=message)
        print(f'[{datetime.now().isoformat()}] Sent "{message}" ({time.thread_time() - t}s)')

    connection.close()


if __name__ == "__main__":
    main()
