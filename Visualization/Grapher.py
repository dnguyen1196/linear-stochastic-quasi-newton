import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import log

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