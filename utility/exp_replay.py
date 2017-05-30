import random
from collections import deque


class ExpReplay(object):
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def append(self, sample):
        self.buffer.append(sample)

    def batch(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        return mini_batch

    def __len__(self):
        return len(self.buffer)
