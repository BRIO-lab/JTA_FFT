import numpy as np
from numba import int32, float32    # import the types
from numba.experimental import jitclass

# Declare array types within this tuple
spec = [
    ('value', int32),               # 'value' is a class parameter which is an int
    ('array', float32[:]),          # 'array' is a class parameter (float 1D array)
]

@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] += val
        return self.array

    @staticmethod
    def add(x, y):
        return x + y

n = 21
mybag = Bag(n)
print(mybag.increment(2))
    