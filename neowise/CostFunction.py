import numpy as np


class Cost:

    @staticmethod
    def binary_cross_entropy(y, A):
        m = y.shape[1]
        cost = (-1 / m) * (np.sum(np.sum((y * np.log(A)) + ((1 - y) * np.log(1 - A)))))
        return cost

    @staticmethod
    def cross_entropy(y, A):
        m = y.shape[1]
        cost = (-1 / m) * (np.sum(np.sum((y * np.log(A)))))
        return cost
