import numpy as np
from neowise.functional import *
from neowise.regularizers import *


class CostFunctionHelpers:
    """
    Cost Function Helpers
    Used to initiate the instance variables of Cost Functions

    Arguments:
        y: Outputs of the train/test data (nd-array)
        A: Activations of the output layer of the network (nd-array)
        layers_arr: A Python list containing objects of nw.layers (list)
        lamb: Regularization parameter "lambda" (float)
        reg: Type of Regularization (str)
    """
    def __init__(self, y, A, layers_arr, lamb, reg=None):
        self.y, self.A, self.layers_arr, self.lamb, self.reg = y, A, layers_arr, lamb, reg


class BinaryCrossEntropy(CostFunctionHelpers):
    """
    Binary Cross Entropy
    Used to calculate the cost for Binary Classification tasks

    Arguments:
        y: Outputs of the train/test data (nd-array)
        A: Activations of the output layer of the network (nd-array)
        layers_arr: A Python list containing objects of nw.layers (list)
        lamb: Regularization parameter "lambda" (float)
        reg: Type of Regularization (str)
    """
    def __init__(self, y, A, layers_arr, lamb, reg=None):
        CostFunctionHelpers.__init__(self, y, A, layers_arr, lamb, reg=None)

    def __call__(self):
        if self.reg is not None:
            if self.reg is "L1":
                GradL1Reg(self.layers_arr)()
                temp_sum = L1Reg(self.layers_arr, self.lamb, self.y.shape[1])()
            if self.reg is "L2":
                temp_sum = L2Reg(self.layers_arr, self.lamb, self.y.shape[1])()
            cost = (-1 / self.y.shape[1]) * (
                np.sum(np.sum((self.y * np.log(self.A)) + ((1 - self.y) * (np.log(1 - self.A)))))) + temp_sum
            grad = (-1 / self.y.shape[1]) * ((self.y / self.A) - ((1 - self.y) / (1 - self.A)))
        else:
            cost = (-1 / self.y.shape[1]) * (
                np.sum(np.sum((self.y * np.log(self.A)) + ((1 - self.y) * (np.log(1 - self.A))))))
            grad = (-1 / self.y.shape[1]) * ((self.y / self.A) - ((1 - self.y) / (1 - self.A)))
            for layers in self.layers_arr:
                layers.grad_reg = 0
        return cost, grad


class CrossEntropy(CostFunctionHelpers):
    """
    Cross Entropy
    Used to calculate the cost for Multi Class Classification tasks

    Arguments:
        y: Outputs of the train/test data (nd-array)
        A: Activations of the output layer of the network (nd-array)
        layers_arr: A Python list containing objects of nw.layers (list)
        lamb: Regularization parameter "lambda" (float)
        reg: Type of Regularization (str)
    """
    def __init__(self, y, A, layers_arr, lamb, reg=None):
        CostFunctionHelpers.__init__(self, y, A, layers_arr, lamb, reg=None)

    def __call__(self):
        if self.reg is not None:
            if self.reg is "L1":
                GradL1Reg(self.layers_arr)()
                temp_sum = L1Reg(self.layers_arr, self.lamb, self.y.shape[1])()
            if self.reg is "L2":
                temp_sum = L2Reg(self.layers_arr, self.lamb, self.y.shape[1])()
            cost = (-1 / self.y.shape[1]) * (np.sum(np.sum((self.y * np.log(self.A)))))
            grad = (-1 / self.y.shape[1]) * ((self.y / self.A))
        else:
            cost = (-1 / self.y.shape[1]) * (np.sum(np.sum((self.y * np.log(self.A)))))
            grad = (-1 / self.y.shape[1]) * ((self.y / self.A))
            for layers in self.layers_arr:
                layers.grad_reg = 0
        return cost, grad
