#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pika


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_publish(exchange='',
                          routing_key='hello',
                          body='Hello world!')
    print('[x] Sent "Hello world!"')
    connection.close()


if __name__ == "__main__":
    main()
