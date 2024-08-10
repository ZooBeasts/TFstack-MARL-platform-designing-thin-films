
from collections import deque


class SharedMemory:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def store(self, design):
        self.buffer.append(design)

    def retrieve(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

class IsolatedMemory:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)
    def store(self, design):
        self.buffer.append(design)

    def retrieve(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

