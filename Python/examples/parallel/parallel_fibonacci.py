#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import threading
import logging
from queue import Queue

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

fibo_dict = {}
shared_queue = Queue()
queue_condition = threading.Condition()
input_list = [3, 10, 5, 7]


def fibonacci_task(condition):
    with condition:
        while shared_queue.empty():
            logger.info("[%s] - waiting for elements in queue.." % threading.current_thread().name)
            condition.wait()
        else:
            value = shared_queue.get()
            a, b = 0, 1
            for item in range(value):
                a, b = b, a + b
                fibo_dict[value] = a
        shared_queue.task_done()
        logger.debug("[%s] - Result %s" % (threading.current_thread().name, fibo_dict))


def queue_task(condition):
    logging.debug("Starting queue_task...")
    with condition:
        for item in input_list:
            shared_queue.put(item)
        logging.debug("Notifying fibonacci_task threads that the queue is ready to consume..")
        condition.notifyAll()


threads = [threading.Thread(daemon=True, target=fibonacci_task, args=(queue_condition,))
           for i in range(4)]
for thread in threads:
    thread.start()

prod = threading.Thread(name='queue_task_thread', daemon=True, target=queue_task, args=(queue_condition,))
prod.start()

for thread in threads:
    thread.join()
