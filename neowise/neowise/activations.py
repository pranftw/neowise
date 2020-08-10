import numpy as np


class Relu:
    def forward(self):
        return np.maximum(0, self)

    def backward(self):
        self[self <= 0] = 0
        self[self > 0] = 1
        return self


class Tanh:
    def forward(self):
        return np.tanh(self)

    def backward(self):
        return (1 - np.square(Tanh.forward(self)))


class Sigmoid:
    def forward(self):
        return (1 / (1 + np.exp(-self)))

    def backward(self):
        return (Sigmoid.forward(self) * (1 - Sigmoid.forward(self)))


class Softmax:
    def forward(self):
        soft = np.exp(self) / np.sum(np.exp(self), axis=0)
        return soft

    def backward(self):
        return (Softmax.forward(self) * (1 - Softmax.forward(self)))


class Sine:
    def forward(self):
        return np.sin(self)

    def backward(self):
        return np.cos(self)
