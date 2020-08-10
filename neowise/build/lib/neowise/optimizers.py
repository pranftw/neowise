import numpy as np


class OptimizerHelpers:
    def __init__(self, alpha, layers_arr, V_dict, S_dict, t):
        self.alpha, self.layers_arr, self.V_dict, self.S_dict, self.t = alpha, layers_arr, V_dict, S_dict, t


class GradientDescent(OptimizerHelpers):
    def __init__(self, alpha, layers_arr, V_dict, S_dict, t):
        OptimizerHelpers.__init__(self, alpha, layers_arr, V_dict, S_dict, t)

    def __call__(self):
        for layers in self.layers_arr:
            layers.weights -= (self.alpha * layers.dW)
            layers.bias -= (self.alpha * layers.db)


class Momentum(OptimizerHelpers):
    def __init__(self, alpha, layers_arr, V_dict, S_dict, t):
        OptimizerHelpers.__init__(self, alpha, layers_arr, V_dict, S_dict, t)

    def __call__(self):
        beta1 = 0.9
        for h in range(1, len(self.layers_arr) + 1):
            self.V_dict["Vdw" + str(h)] = (beta1 * self.V_dict["Vdw" + str(h)]) + (
                    (1 - beta1) * self.layers_arr[h - 1].dW)
            self.V_dict["Vdb" + str(h)] = (beta1 * self.V_dict["Vdb" + str(h)]) + (
                    (1 - beta1) * self.layers_arr[h - 1].db)
        for g in range(1, len(self.layers_arr) + 1):
            self.layers_arr[g - 1].weights -= (self.alpha * self.V_dict["Vdw" + str(g)])
            self.layers_arr[g - 1].bias -= (self.alpha * self.V_dict["Vdb" + str(g)])


class RMSProp(OptimizerHelpers):
    def __init__(self, alpha, layers_arr, V_dict, S_dict, t):
        OptimizerHelpers.__init__(self, alpha, layers_arr, V_dict, S_dict, t)

    def __call__(self):
        beta2 = 0.999
        epsilon = 1e-8
        for h in range(1, len(self.layers_arr) + 1):
            self.S_dict["Sdw" + str(h)] = (beta2 * self.S_dict["Sdw" + str(h)]) + (
                    (1 - beta2) * np.square(self.layers_arr[h - 1].dW))
            self.S_dict["Sdb" + str(h)] = (beta2 * self.S_dict["Sdb" + str(h)]) + (
                    (1 - beta2) * np.square(self.layers_arr[h - 1].db))
        for g in range(1, len(self.layers_arr) + 1):
            self.layers_arr[g - 1].weights -= (
                    (self.alpha * self.layers_arr[g - 1].dW) / (np.sqrt(self.S_dict["Sdw" + str(g)]) + epsilon))
            self.layers_arr[g - 1].bias -= (
                    (self.alpha * self.layers_arr[g - 1].db) / (np.sqrt(self.S_dict["Sdb" + str(g)]) + epsilon))


class Adam(OptimizerHelpers):
    def __init__(self, alpha, layers_arr, V_dict, S_dict, t):
        OptimizerHelpers.__init__(self, alpha, layers_arr, V_dict, S_dict, t)

    def __call__(self):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        S_dict_corr = {}
        V_dict_corr = {}
        for h in range(1, len(self.layers_arr) + 1):
            self.V_dict["Vdw" + str(h)] = (beta1 * self.V_dict["Vdw" + str(h)]) + (
                    (1 - beta1) * self.layers_arr[h - 1].dW)
            self.V_dict["Vdb" + str(h)] = (beta1 * self.V_dict["Vdb" + str(h)]) + (
                    (1 - beta1) * self.layers_arr[h - 1].db)
        for u in range(1, len(self.layers_arr) + 1):
            self.S_dict["Sdw" + str(u)] = (beta2 * self.S_dict["Sdw" + str(u)]) + (
                    (1 - beta2) * np.square(self.layers_arr[u - 1].dW))
            self.S_dict["Sdb" + str(u)] = (beta2 * self.S_dict["Sdb" + str(u)]) + (
                    (1 - beta2) * np.square(self.layers_arr[u - 1].db))
        for n in range(1, len(self.layers_arr) + 1):
            S_dict_corr["Sdw" + str(n)] = self.S_dict["Sdw" + str(n)] / (1 - np.power(beta2, self.t))
            S_dict_corr["Sdb" + str(n)] = self.S_dict["Sdb" + str(n)] / (1 - np.power(beta2, self.t))
            V_dict_corr["Vdw" + str(n)] = self.V_dict["Vdw" + str(n)] / (1 - np.power(beta1, self.t))
            V_dict_corr["Vdb" + str(n)] = self.V_dict["Vdb" + str(n)] / (1 - np.power(beta1, self.t))
        for g in range(1, len(self.layers_arr) + 1):
            self.layers_arr[g - 1].weights -= (
                    (self.alpha * V_dict_corr["Vdw" + str(g)]) / (np.sqrt(S_dict_corr["Sdw" + str(g)]) + epsilon))
            self.layers_arr[g - 1].bias -= (
                    (self.alpha * V_dict_corr["Vdb" + str(g)]) / (np.sqrt(S_dict_corr["Sdb" + str(g)]) + epsilon))
