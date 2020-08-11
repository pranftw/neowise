from neowise.activations import *


class Dense:
    """
    Dense
    Creates Fully connected Neural Network layer

    Arguments:
         num_inputs: Number of input units to the current layer (int)
         num_outputs: Number of units in the current layer (int)
         activation_fn: Activation function for the current layer (str)
         dropout: Value between 0 and 1 (default=1) (float)
         weights: Weights for the layer in dimensions (num_outputs,num_inputs) (nd-array)
         bias: Bias of the layer in dimensions (num_outputs,1) (nd-array)
         dZ: Gradient of Cost function w.r.t the outputs of the layer (nd-array)
         dW: Gradient of Cost function w.r.t the weights of the layer (nd-array)
         db: Gradient of Cost function w.r.t the bias of the layer (nd-array)
         dA: Gradient of Cost function w.r.t the activations of the layer (nd-array)
         grad_L1: Gradient of the weights w.r.t itself (nd-array)
         grad_reg: Calculates the gradient of Regularization functions (nd-array)
    """
    def __init__(self, num_inputs, num_outputs, activation_fn, dropout=1.0, weights=None, bias=None, dZ=None, dW=None,
                 db=None, dA=None, grad_L1=None, grad_reg=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.dZ, self.dW, self.db, self.dA = dZ, dW, db, dA
        self.grad_L1 = grad_L1
        self.grad_reg = grad_reg
        self.weights = weights
        self.bias = bias
        self.activ_dict = {"relu": [Relu.forward, Relu.backward, 2],
                           "tanh": [Tanh.forward, Tanh.backward, 1],
                           "sigmoid": [Sigmoid.forward, Sigmoid.backward, 1],
                           "softmax": [Softmax.forward, Softmax.backward, 1],
                           "sine": [Sine.forward, Sine.backward, 6]}

    def initialize_params(self):
        self.weights = np.random.randn(self.num_outputs, self.num_inputs) * (
            np.sqrt(self.activ_dict[self.activation_fn][2] / self.num_inputs))
        self.bias = np.random.randn(self.num_outputs, 1) * 0.01
        return self.weights, self.bias, self.grad_reg, self.grad_L1

    def get_params(self):
        return self.weights, self.bias

    def forw_prop(self, A_prev, train=True):
        if train is False:
            self.dropout = 1
        self.outputs = np.dot(self.weights, A_prev) + self.bias
        self.activations_temp = self.activ_dict[self.activation_fn][0](self.outputs)
        self.activations = self.activations_temp * (
                (np.random.rand(self.outputs.shape[0], self.outputs.shape[1]) < self.dropout) / self.dropout)
        return self.outputs, self.activations

    def back_prop(self, dA_prev, A_prev):
        self.dZ = dA_prev * self.activ_dict[self.activation_fn][1](self.outputs)
        self.dW = (np.dot(self.dZ, A_prev.T)) + self.grad_reg
        self.db = np.sum(self.dZ, axis=1, keepdims=True)
        self.dA = np.dot(self.weights.T, self.dZ)
        return self.dZ, self.dW, self.db, self.dA
