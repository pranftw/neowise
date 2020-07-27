import numpy as np


class Parameters:
    def __init__(self, num_inputs, num_outputs, activation_fn):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation_fn = activation_fn
        self.weights = None
        self.bias = None
        if (self.activation_fn == "relu") or (self.activation_fn == "sigmoid"):
            self.initi = 2
        if (self.activation_fn == "tanh") or (self.activation_fn == "softmax"):
            self.initi = 1

    def Weights(self):
        self.weights = np.random.randn(self.num_outputs, self.num_inputs) * (np.sqrt(self.initi / self.num_inputs))

    def Bias(self):
        self.bias = np.random.randn(self.num_outputs) * 0.01
