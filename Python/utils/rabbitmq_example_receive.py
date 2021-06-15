#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json

import pika


def callback(channel, method, properties, body):
    body = json.loads(body)
    print(f'[x] Received: {body}')


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(queue='hello',
                          auto_ack=True,
                          on_message_callback=callback)
    print('[*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
