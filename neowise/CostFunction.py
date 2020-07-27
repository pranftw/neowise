import numpy as np


class Cost:

    def binary_cross_entropy(self, y, A, m):
        cost = (-1 / m) * (np.sum(np.sum((y * np.log(A)) + ((1 - y) * np.log(1 - A)))))
        return cost

    def cross_entropy(self, y, A, m):
        cost = (-1 / m) * (np.sum(np.sum((y * np.log(A)))))
        return cost
