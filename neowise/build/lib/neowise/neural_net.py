from neowise.activations import *
from neowise.cost_function import *
from neowise.functional import *
from neowise.layers import *
from neowise.optimizers import *
from neowise.plots import *
from tqdm import tqdm
import sys
import time
import hdfdict
from prettytable import PrettyTable


class Model:
    def __init__(self, X_tr, y_tr, X_te, y_te, X_cv, y_cv):
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_te, self.y_te = X_te, y_te
        self.X_cv, self.y_cv = X_cv, y_cv
        self.layer_names = []
        self.layer_names_arr = []
        self.activations_cache = None
        self.params = None
        self.arch = {}
        self.accu_tr_arr = None
        self.cost_tr_arr = None
        self.accu_cv_arr = None
        self.cost_cv_arr = None
        self.lr = None
        self.epochs = None

    def add(self, layer_name, num_inputs, num_outputs, act_fn, dropout=1):
        self.layer_names_arr.append(layer_name)
        self.arch[str(layer_name)] = [layer_name.encode('utf8'), num_inputs, num_outputs, act_fn.encode('utf8'),
                                      dropout]
        layer_name = Dense(num_inputs, num_outputs, act_fn, dropout)
        Dense.initialize_params(layer_name)
        self.layer_names.append(layer_name)

    def reset(self):
        self.layer_names = []
        self.arch = {}
        self.layer_names_arr = []
        return self.layer_names, self.arch, self.layer_names_arr

    def params_dict(self, print_params):
        self.params = {}
        hee = 1
        for layer in self.layer_names:
            self.params["W" + str(hee)], self.params["b" + str(hee)] = Dense.get_params(layer)
            hee += 1
        if print_params is True:
            print(self.params)
            return self.params
        else:
            return self.params

    def forward_prop(self, X, train_model=True):
        self.activations_cache = {}
        self.activations_cache = {"A0": X.T}
        temp_A = X.T
        p = 1
        for layer in self.layer_names:
            _, temp_A = Dense.forw_prop(layer, temp_A, train_model)
            self.activations_cache["A" + str(p)] = temp_A
            p += 1
        return self.activations_cache

    def backward_prop(self, y, prob_type, activations_cache, lamb, reg):
        prob_type_dict = {"Binary": [BinaryCrossEntropy, PrecisionRecall, Predict, Evaluate],
                          "Multi": [CrossEntropy, PrecisionRecallMulti, PredictMulti, EvaluateMulti]}
        _, temp_dA = prob_type_dict[prob_type][0](y, activations_cache["A" + str(len(self.layer_names))],
                                                  self.layer_names, lamb, reg)()
        l = 1
        for layer in reversed(self.layer_names):
            _, layer.dW, layer.db, temp_dA = Dense.back_prop(layer, temp_dA, self.activations_cache[
                "A" + str(len(self.layer_names) - l)])
            l += 1

    def fit(self, X, y, alpha, num_iter, optim, prob_type, mb, reg=None, lamb=None, alpha_decay=False, print_cost=True,
            callback=None):
        self.lr = alpha
        if (num_iter > 100) and (num_iter <= 1000):
            freq = 10
        elif num_iter > 1000:
            freq = 50
        elif (num_iter <= 100):
            freq = 1
        params = self.params_dict(print_params=False)
        V_dict = {}
        S_dict = {}
        mini_batches, num = CreateMiniBatches(X, y, mb)()
        self.accu_tr_arr = []
        self.cost_tr_arr = []
        self.accu_cv_arr = []
        self.cost_cv_arr = []
        for k in range(1, len(self.layer_names) + 1):
            V_dict["Vdw" + str(k)] = np.zeros(params["W" + str(k)].shape)
            V_dict["Vdb" + str(k)] = np.zeros(params["b" + str(k)].shape)
            S_dict["Sdw" + str(k)] = np.zeros(params["W" + str(k)].shape)
            S_dict["Sdb" + str(k)] = np.zeros(params["b" + str(k)].shape)
        optim_dict = {"GD": [GradientDescent, None, None, 0],
                      "Momentum": [Momentum, V_dict, None, 0],
                      "RMSprop": [RMSProp, None, S_dict, 0],
                      "Adam": [Adam, V_dict, S_dict, 0]}
        prob_type_dict = {"Binary": [BinaryCrossEntropy, PrecisionRecall, Predict, Evaluate],
                          "Multi": [CrossEntropy, PrecisionRecallMulti, PredictMulti, EvaluateMulti]}

        for i in range(1, num_iter + 1):
            params_plot = self.params_dict(print_params=False)
            if alpha_decay is True:
                alpha = (np.power(0.95, i)) * self.lr
            for vee in tqdm(range(0, num + 1), file=sys.stdout):
                activations_dict = self.forward_prop(mini_batches["MB_X" + str(vee)])
                self.backward_prop(mini_batches["MB_Y" + str(vee)], prob_type, activations_dict, lamb, reg)
                optim_dict[optim][0](alpha, self.layer_names, optim_dict[optim][1], optim_dict[optim][2],
                                     optim_dict[optim][3] + i)()
            act_tr = self.forward_prop(X)
            cost_tr, _ = prob_type_dict[prob_type][0](y, act_tr["A" + str(len(self.layer_names))], self.layer_names,
                                                      lamb, reg)()
            preds = prob_type_dict[prob_type][2](act_tr["A" + str(len(self.layer_names))])()
            accu_tr = prob_type_dict[prob_type][3](y, preds)()
            self.accu_tr_arr.append(accu_tr)
            self.cost_tr_arr.append(cost_tr)
            if self.X_cv is not None:
                act_cv = self.forward_prop(self.X_cv)
                cost_cv, _ = prob_type_dict[prob_type][0](self.y_cv, act_cv["A" + str(len(self.layer_names))],
                                                          self.layer_names, lamb, reg)()
                preds_cv = prob_type_dict[prob_type][2](act_cv["A" + str(len(self.layer_names))])()
                accu_cv = prob_type_dict[prob_type][3](self.y_cv, preds_cv)()
                self.accu_cv_arr.append(accu_cv)
                self.cost_cv_arr.append(cost_cv)
            if ((i % 1 == 0) and print_cost == True):
                if self.X_cv is None:
                    print("Iteration " + str(i) + " " + "train_cost: " + str(
                        np.round(cost_tr, 6)) + " --- " + "train_acc: " + str(np.round(accu_tr, 3)))
                else:
                    print("Iteration " + str(i) + " " + "train_cost: " + str(
                        np.round(cost_tr, 6)) + " --- " + "train_acc: " + str(
                        np.round(accu_tr, 3)) + " --- " + "val_cost: " + str(
                        np.round(cost_cv, 6)) + " --- " + "val_accu: " + str(np.round(accu_cv, 3)))
            if (i % 1 == 0):
                if (callback is not None):
                    callback(i, params_plot)

    def save_model(self, fname):
        params = self.params_dict(print_params=False)
        archi = self.arch
        model_dict = {"Parameters": params, "Architecture": archi}
        hdfdict.dump(model_dict, fname)
        print("Model saved!")

    def load_model(self, fname):
        print("Model loading....")
        model_dict = dict(hdfdict.load(fname))
        params_dict = model_dict["Parameters"]
        arch_dict = model_dict["Architecture"]
        self.reset()
        for key in arch_dict:
            self.add(arch_dict[key][0].decode('utf8'), int(arch_dict[key][1]), int(arch_dict[key][2]),
                     arch_dict[key][3].decode('utf8'), int(arch_dict[key][4]))
        dee = 1
        for layer in self.layer_names:
            layer.weights = params_dict["W" + str(dee)]
            layer.bias = params_dict["b" + str(dee)]
            dee += 1
        print("Model loaded!")

    def plot(self, type_func, animate=False, direc=None):
        itera = np.arange(1, len(self.cost_tr_arr) + 1)
        if (len(itera) > 100) and (len(itera) <= 1000):
            freq = 10
        elif len(itera) > 1000:
            freq = 50
        elif (len(itera) <= 100):
            freq = 1
        plot_dict = {"Cost": [PlotCostStatic, AnimatePlotMulti, self.cost_tr_arr, self.cost_cv_arr, 'Costs',
                              'Train-Cross Val Costs Curve', 'upper right', ['c', '#9ef705'], AnimatePlot, "Costs",
                              "Cost Function Curve"],
                     "Accuracy": [PlotTrCvStatic, AnimatePlotMulti, self.accu_tr_arr, self.accu_cv_arr, 'Accuracy',
                                  'Train-Cross Val Accuracy Curve', 'lower right', ['m', 'r'], AnimatePlot, "Accuracy",
                                  "Accuracy Curve"]}
        if animate is False:
            plot_dict[type_func][0](self.cost_tr_arr, self.cost_cv_arr, self.accu_tr_arr, self.accu_cv_arr)()
        else:
            if len(self.cost_cv_arr) != 0:
                plot_dict[type_func][1]([itera, itera], [plot_dict[type_func][2], plot_dict[type_func][3]],
                                        'Number of Iterations', plot_dict[type_func][4], plot_dict[type_func][5],
                                        ['Train', 'Cross Val'], plot_dict[type_func][6], plot_dict[type_func][7], direc,
                                        freq)()
            else:
                plot_dict[type_func][8](itera, plot_dict[type_func][2], "Number of Iterations", plot_dict[type_func][9],
                                        plot_dict[type_func][10], "Train", plot_dict[type_func][6],
                                        plot_dict[type_func][7][0], direc, freq)()
            print("Go to your directory to find the images! Feed them to a GIF creator to animate them!")

    def summary(self):
        tab = PrettyTable()
        tab.field_names = ["Layer Number", "Layer Name", "Inputs", "Outputs", "Activation", "Dropout",
                           "Number of Parameters"]
        yee = 1
        total_params = 0
        for layer in self.layer_names:
            total_params += (layer.weights.shape[0] * layer.weights.shape[1]) + len(layer.bias)
            tab.add_row([str(yee), self.layer_names_arr[yee - 1], layer.weights.shape[1], layer.weights.shape[0],
                         layer.activation_fn, layer.dropout,
                         str((layer.weights.shape[0] * layer.weights.shape[1]) + len(layer.bias))])
            yee += 1
        print(tab)
        print("Total number of trainable Parameters: " + str(total_params))

    def test(self, X, y, prob_type, training=False, print_values=True):
        prob_type_dict = {"Binary": [BinaryCrossEntropy, PrecisionRecall, Predict, Evaluate],
                          "Multi": [CrossEntropy, PrecisionRecallMulti, PredictMulti, EvaluateMulti]}
        act_te = self.forward_prop(X, training)
        predictions_te = prob_type_dict[prob_type][2](act_te["A" + str(len(self.layer_names))])()
        accu_te = prob_type_dict[prob_type][3](y, predictions_te)()
        prec_te, rec_te, f1_te = prob_type_dict[prob_type][1](predictions_te, y)()
        tab = PrettyTable()
        if prob_type == "Multi":
            tab.field_names = ["Class", "Precision", "Recall", "F1"]
            for hee in range(0, y.shape[0]):
                tab.add_row([hee, prec_te["class" + str(hee)], rec_te["class" + str(hee)], f1_te["class" + str(hee)]])
            if print_values is True:
                print(tab)
                print("Test Accuracy: " + str(accu_te))
        if prob_type == "Binary":
            tab.field_names = ["Precision", "Recall", "F1"]
            tab.add_row([str(prec_te), str(rec_te), str(f1_te)])
            if print_values is True:
                print(tab)
                print("Test Accuracy: " + str(accu_te))
        if print_values is not True:
            return accu_te, prec_te, rec_te, f1_te
