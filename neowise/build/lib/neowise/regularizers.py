import numpy as np


class RegularizationHelpers:
    def __init__(self, layers_arr, lamb, m_exam):
        self.layers_arr, self.lamb, self.m_exam = layers_arr, lamb, m_exam


class L1Reg(RegularizationHelpers):
    def __init__(self, layers_arr, lamb, m_exam):
        RegularizationHelpers.__init__(self, layers_arr, lamb, m_exam)

    def __call__(self):
        temp_sum = 0
        for layers in self.layers_arr:
            temp_sum = temp_sum + ((self.lamb / self.m_exam) * (np.sum(np.sum(layers.weights))))
            layers.grad_reg = ((self.lamb / self.m_exam) * (layers.grad_L1))
        return temp_sum


class L2Reg(RegularizationHelpers):
    def __init__(self, layers_arr, lamb, m_exam):
        RegularizationHelpers.__init__(self, layers_arr, lamb, m_exam)

    def __call__(self):
        temp_sum = 0
        for layers in self.layers_arr:
            temp_sum = temp_sum + ((self.lamb / (2 * self.m_exam)) * (np.sum(np.sum(np.square(layers.weights)))))
            layers.grad_reg = ((self.lamb / self.m_exam) * (layers.weights))
        return temp_sum
