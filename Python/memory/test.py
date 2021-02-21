#!/usr/bin/python3
# -*- coding: utf-8 -*-
import heapq
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


class Heap:

    def __init__(self):
        self._data = []
        self._key_data = []
        self._index = 0

    def push(self, item):
        self._index += 1
        heapq.heappush(self._data, item)
        item = (self.key(item), self._index, item)
        heapq.heappush(self._key_data, item)

    def key(self, item: float):
        return -1 * item

    @property
    def data(self):
        return self._data.copy()

    @property
    def key_data(self):
        return self._key_data.copy()


if __name__ == "__main__":
    heap = Heap()
    for _ in range(16):
        heap.push(random.randint(0, 1000))

    #
    #print(f'Heap: {heap.data}')
    #print(f'Heap: {heap.key_data}')

    a = []
    b = []
    while len(heap.data) > 0:
        a.append(heapq.heappop(heap._data))

    while len(heap.key_data) > 0:
        b.append(heapq.heappop(heap._key_data))

    for ai, bi in zip(a, b):
        print(ai, bi)
