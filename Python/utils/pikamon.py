#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import inspect

import pika


class Publisher:

    def __init__(self, host='localhost', port=5672, queue='navops'):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port))
        self._channel = connection.channel()
        self._queue = queue
        self.channel.queue_declare(queue=self.queue)

    def enqueue(self, message, exchange=''):
        message = self._message_middleware(message)
        self.channel.basic_publish(exchange=exchange,
                                   routing_key=self.queue,
                                   body=message)

    def _message_middleware(self, message):
        if type(message) is str:
            try:
                _ = json.loads(message)
            except json.decoder.JSONDecodeError:
                raise json.decoder.JSONDecodeError()
        elif type(message) is dict:
            message = json.dumps(message)
        return message

    @property
    def channel(self):
        return self._channel

    @property
    def queue(self):
        return self._queue


class Subscriber:

    def __init__(self, host='localhost', port=5672, queue='navops'):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', port=port))
        self._channel = connection.channel()
        self._queue = queue
        self.channel.queue_declare(queue=self.queue)
        self.channel.basic_consume(
            queue=self.queue,
            auto_ack=True,
            on_message_callback=self.on_message_callback)
        self._pipelines = []

    def run(self):
        self.channel.start_consuming()

    def on_message_callback(self, channel, method, properties, body):
        message = json.loads(body)
        _ = self._pipeline(message)

    def add_pipeline(self, func):
        if self._is_pipeline_compatible(func):
            self._pipelines.append(func)

    def remove_pipeline(self, func):
        if func in self._pipelines:
            self._pipelines.remove(func)

    def _pipeline(self, message):
        for pipeline_fn in self._pipelines:
            message = pipeline_fn(message)
        return message

    def _is_pipeline_compatible(self, func) -> bool:
        if not callable(func):
            return False
        argspec = inspect.getfullargspec(func)
        # if len(argspec) != 1 and argspec.annotations.get('return', str) is not str
        return (len(argspec.args) - ('self' in argspec.args)) == 1

    @property
    def channel(self):
        return self._channel

    @property
    def queue(self):
        return self._queue


if __name__ == "__main__":
    pass
