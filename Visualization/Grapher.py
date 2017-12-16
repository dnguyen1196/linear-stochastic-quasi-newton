import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import log
import numpy as np


def plot_compare_log_w_norm(series1, series2, title1, title2):
    #
    series1 = series1[~np.isnan(series1)]
    series2 = series2[~np.isnan(series2)]

    w_final_1 = series1[-1]
    w_final_2 = series2[-1]
    log_norm_1 = []
    for i in range(len(series1)-1):
        res = series1[i] - w_final_1
        log_norm_1.append(log(norm(res)))

    log_norm_2 = []
    for i in range(len(series2)-1):
        res = series2[i] - w_final_2
        log_norm_2.append(log(norm(res)))

    plt.plot(log_norm_1, "r--", label=title1)
    plt.plot(log_norm_2, 'b--', label=title2)
    plt.ylabel("log|W-W*|")
    plt.xlabel("iterations")
    plt.legend(loc='upper right')
    plt.show()


def plot_compare_log_f_norm(series1, series2, title1, title2):
    series1 = [val for val in series1 if ~np.any(np.isnan(val))]
    series2 = [val for val in series2 if ~np.any(np.isnan(val))]

    f_final_1 = series1[-1]
    f_final_2 = series2[-1]
    log_norm_1 = []
    for i in range(len(series1)-1):
        res = series1[i] - f_final_1
        log_norm_1.append(log(norm(res)))

    log_norm_2 = []
    for i in range(len(series2)-1):
        res = series2[i] - f_final_2
        log_norm_2.append(log(norm(res)))

    max_iters = max(len(log_norm_1), len(log_norm_2))
    plt.plot(log_norm_1, "r--", label=title1)
    plt.plot(log_norm_2, 'b--', label=title2)
    plt.ylabel("log|f-f*|")
    plt.xlabel("iterations")
    plt.legend(loc='upper right')
    plt.show()


def plot_w_norm(w_):
    w_final = w_[-1]
    log_norm = []
    for i in range(len(w_)-1):
        res = w_[i] - w_final
        log_norm.append(log(norm(res)))
    plt.plot(log_norm)
    plt.ylabel("log|W-W*|")
    plt.xlabel("iterations")
    plt.show()


def plot_f_norm(f_):
    f_final = f_[-1]
    log_norm = []
    for i in range(len(f_)-1):
        res = f_[i] - f_final
        log_norm.append(log(res))
    plt.plot(log_norm)
    plt.ylabel("log(f-f*)")
    plt.xlabel("iterations")
    plt.show()