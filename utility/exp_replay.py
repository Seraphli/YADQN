import random
from collections import deque


class ExpReplay(object):
    def __init__(self, size):
        self.size = size
        self._buffer = deque(maxlen=size)

    def add(self, s, a, r, t, s_):
        self._buffer.append((s, a, r, t, s_))

    def batch(self, batch_size):
        idxes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        s, a, r, t, s_ = [], [], [], [], []
        for i in idxes:
            s.append(self._buffer[i][0])
            a.append(self._buffer[i][1])
            r.append(self._buffer[i][2])
            t.append(self._buffer[i][3])
            s_.append(self._buffer[i][4])
        return s, a, r, t, s_

    def __len__(self):
        return len(self._buffer)
