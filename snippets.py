# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-04 2021
'''
import numpy as np


class BaseGenerator(object):
    """数据生成器模版"""

    def __init__(self, data, batch_size, buffer_size, random, mode='train'):
        self.data = data
        self.mode = mode
        self.batch_size = batch_size
        if hasattr(self.data, "__len__"):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or self.batch_size * 1000
        self.random = random

    def __len__(self):
        return self.steps

    def __iter__(self):
        raise NotImplementedError

    def generate(self):
        while True:
            for d in self.__iter__():
                yield d

    def sample(self):
        """采样函数，每个样本同时返回一个is_end标记"""
        if self.random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        current_data = next(data)
        for next_data in data:
            yield False, current_data
            current_data = next_data

        yield True, current_data
