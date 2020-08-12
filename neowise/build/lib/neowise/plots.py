import numpy as np
import matplotlib.pyplot as plt


class StaticPlotHelpers:
    """
    Static Plot Helpers
    Used to initiate the instance variables of Static Plots

    Arguments:
        costs_tr: Array of costs of training data through the entire training duration of the network
        costs_cv: Array of costs of cross validation data through the entire duration of the network
        accu_tr_arr: Array containing the accuracy of the training data of the network through the entire training duration
        accu_cv_arr: Array containing the accuracy of the cross  validation data of the network through the entire training duration
    """
    def __init__(self, costs_tr, costs_cv, accu_tr_arr, accu_cv_arr):
        self.costs_tr, self.costs_cv = costs_tr, costs_cv
        self.accu_tr_arr, self.accu_cv_arr = accu_tr_arr, accu_cv_arr


class AnimatePlotHelpers:
    """
    Animate Plot Helpers
    Used to initiate the instance variables of Animated Plots

    Arguments:
        x_ax: X axis data for plotting
        y_ax: Y axis data for plotting
        X_lab: Label of the X axis of the plot
        Y_lab: Label of the Y axis of the plot
        plot_title: Title of the plot
        leg: Legend of the plot
        loca: Location of legends
        plot_col: Color of the plotted line
        direc: Directory in which the images should be saved
        freq: Frequency with which the plot should update
    """
    def __init__(self, x_ax, y_ax, X_lab, Y_lab, plot_title, leg, loca, plot_col, direc, freq):
        self.x_ax, self.y_ax, self.X_lab, self.Y_lab = x_ax, y_ax, X_lab, Y_lab
        self.plot_title, self.leg, self.loca, self.plot_col, self.direc, self.freq = plot_title, leg, loca, plot_col, direc, freq


class PlotCostStatic(StaticPlotHelpers):
    """
    Plot Cost Static
    Plots the Cost vs Number of iterations Plot

    Arguments:
        costs_tr: Array of costs of training data through the entire training duration of the network
        costs_cv: Array of costs of cross validation data through the entire duration of the network
        accu_tr_arr: Array containing the accuracy of the training data of the network through the entire training duration
        accu_cv_arr: Array containing the accuracy of the cross  validation data of the network through the entire training duration
    """
    def __init__(self, costs_tr, costs_cv, accu_tr_arr, accu_cv_arr):
        StaticPlotHelpers.__init__(self, costs_tr, costs_cv, accu_tr_arr, accu_cv_arr)

    def __call__(self):
        itera = np.arange(1, len(self.costs_tr) + 1, 1)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function Variation')
        plt.plot(itera, self.costs_tr, color='c', linewidth=2)
        if len(self.costs_cv) != 0:
            plt.plot(itera, self.costs_cv, color='#9ef705', linewidth=2)
        plt.legend(["Train", "Cross Val"], loc='upper right')


class PlotTrCvStatic(StaticPlotHelpers):
    """
    Plot Train CrossVal Static
    Plots the accuracies of train and cross val vs number of iterations

    Arguments:
        costs_tr: Array of costs of training data through the entire training duration of the network
        costs_cv: Array of costs of cross validation data through the entire duration of the network
        accu_tr_arr: Array containing the accuracy of the training data of the network through the entire training duration
        accu_cv_arr: Array containing the accuracy of the cross  validation data of the network through the entire training duration
    """
    def __init__(self, costs_tr, costs_cv, accu_tr_arr, accu_cv_arr):
        StaticPlotHelpers.__init__(self, costs_tr, costs_cv, accu_tr_arr, accu_cv_arr)

    def __call__(self):
        itera = np.arange(1, len(self.costs_tr) + 1, 1)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Accuracy')
        plt.title('Train - Cross Val Accuracy Curve')
        plt.plot(itera, self.accu_tr_arr, color='m', linewidth=2)
        if len(self.accu_cv_arr) != 0:
            plt.plot(itera, self.accu_cv_arr, color='r', linewidth=2)
        plt.legend(["Train", "Cross Val"], loc='lower right')


class AnimatePlot(AnimatePlotHelpers):
    """
    Animate Plot
    Animates the plot which has only one argument for y axis

    Arguments:
        x_ax: X axis data for plotting
        y_ax: Y axis data for plotting
        X_lab: Label of the X axis of the plot
        Y_lab: Label of the Y axis of the plot
        plot_title: Title of the plot
        leg: Legend of the plot
        loca: Location of legends
        plot_col: Color of the plotted line
        direc: Directory in which the images should be saved
        freq: Frequency with which the plot should update
    """
    def __init__(self, x_ax, y_ax, X_lab, Y_lab, plot_title, leg, loca, plot_col, direc, freq):
        AnimatePlotHelpers.__init__(self, x_ax, y_ax, X_lab, Y_lab, plot_title, leg, loca, plot_col, direc, freq)

    def __call__(self):
        y_vals = self.y_ax
        x_vals = self.x_ax
        l = 0
        for k in range(1, len(x_vals) + 1):
            plt.xlabel(self.X_lab)
            plt.ylabel(self.Y_lab)
            plt.title(self.plot_title)
            plt.legend([self.leg], loc=self.loca)
            plt.plot(x_vals[0:l], y_vals[0:l], color=self.plot_col)
            l = l + 1
            if k % self.freq == 0:
                plt.savefig(self.direc + 'plot{}.png'.format(k // self.freq))
        return


class AnimatePlotMulti(AnimatePlotHelpers):
    """
    Animate Plot Multi
    Animates the plot with more than one arguments for Y axis

    Arguments:
        x_ax: X axis data for plotting
        y_ax: Y axis data for plotting
        X_lab: Label of the X axis of the plot
        Y_lab: Label of the Y axis of the plot
        plot_title: Title of the plot
        leg: Legend of the plot
        loca: Location of legends
        plot_col: Color of the plotted line
        direc: Directory in which the images should be saved
        freq: Frequency with which the plot should update
    """
    def __init__(self, x_ax, y_ax, X_lab, Y_lab, plot_title, leg, loca, plot_col, direc, freq):
        AnimatePlotHelpers.__init__(self, x_ax, y_ax, X_lab, Y_lab, plot_title, leg, loca, plot_col, direc, freq)

    def __call__(self):
        x_vals = {}
        y_vals = {}
        for m in range(0, len(self.x_ax)):
            x_vals["X" + str(m)] = self.x_ax[m]
        for g in range(0, len(self.y_ax)):
            y_vals["Y" + str(g)] = self.y_ax[g]
        for h in range(0, len(self.leg)):
            plt.plot([], [], color=self.plot_col[h], label=self.leg[h])
        plt.legend(loc=self.loca)
        l = 0
        for k in range(1, len(self.x_ax[0]) + 1):
            plt.xlabel(self.X_lab)
            plt.ylabel(self.Y_lab)
            plt.title(self.plot_title)
            for d in range(0, len(self.x_ax)):
                if len(x_vals["X" + str(d)]) != 0:
                    plt.plot(x_vals["X" + str(d)][0:l], y_vals["Y" + str(d)][0:l], color=self.plot_col[d])
            l = l + 1
            if k % self.freq == 0:
                plt.savefig(self.direc + 'plot{}.png'.format(k // self.freq))
        return
