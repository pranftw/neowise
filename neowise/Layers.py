from neowise import ActivationFunction, InitializeParameters
import numpy as np


class Layer:
    pass


class Dense:
    def __init__(self, layer_num, num_inputs, num_outputs, activation_fn):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation_fn = activation_fn
        self.layer_num = layer_num
        self.output = None
        self.activation = None
        self.weights = InitializeParameters.Parameters(self.num_inputs, self.num_outputs, str(activation_fn)).Weights()
        self.bias = InitializeParameters.Parameters(self.num_inputs, self.num_outputs, str(activation_fn)).Bias()
        self.dA = None
        self.dZ = None
        self.dW = None
        self.db = None

    def forw_prop(self, A_prev):
        self.output = np.dot(self.weights, A_prev) + self.bias
        if self.activation_fn == "relu":
            self.activation = ActivationFunction.ActivationsForward.relu(self.output)
        if self.activation_fn == "sigmoid":
            self.activation = ActivationFunction.ActivationsForward.sigmoid(self.output)
        if self.activation_fn == "tanh":
            self.activation = ActivationFunction.ActivationsForward.tanh(self.output)
        if self.activation_fn == "softmax":
            self.activation = ActivationFunction.ActivationsForward.softmax(self.output)
