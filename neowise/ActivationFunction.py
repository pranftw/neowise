import numpy as np


class ActivationsForward:

    def relu(self):
        return np.maximum(0, self)

    def sigmoid(self):
        return 1 / (1 + (np.exp(self)*np.exp(-1)))

    def softmax(self):
        soft = np.exp(self) / np.sum(np.exp(self), axis=0)
        return soft

    def tanh(self):
        return np.tanh(self)


class ActivationsBackward:
    def relu_back(self):
        self[self <= 0] = 0
        self[self > 0] = 1
        return self

    def sigmoid_back(self):
        grad_back = (np.exp(self)*np.exp(-1)) / np.square(1 + (np.exp(self)*np.exp(-1)))
        return grad_back

    def tanh_back(self):
        return 1 - np.square(np.tanh(self))




