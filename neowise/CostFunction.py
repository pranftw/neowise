import numpy as np


class Cost:
    def __init__(self, y, A, m):
        self.y = y
        self.A = A
        self.m = m

    def binary_cross_entropy(self):
        cost = (-1 / self.m) * (np.sum(np.sum((self.y * np.log(self.A)) + ((1 - self.y) * np.log(1 - self.A)))))
        return cost

    def cross_entropy(self):
        cost = (-1 / self.m) * (np.sum(np.sum((self.y * np.log(self.A)))))
        return cost
