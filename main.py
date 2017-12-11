import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from ObjectiveFunc.Quadratic import QuadraticFunction
from ObjectiveFunc.Logistic import LogisticRegression
from SQN.LinearSQN import StochasticLBFGS
from Visualization import Grapher as graphing

N = 400
n = 2

quadratic_cost = QuadraticFunction(N=N, n=n)
logistic_cost = LogisticRegression(N=N, n=n)

S_LBFGS = StochasticLBFGS(obj=quadratic_cost)
S_LBFGS_log = StochasticLBFGS(obj=logistic_cost)

# w_lbfgs_his, f_lbfgs_his = S_LBFGS_log.optimize()
w_lbfgs_his, f_lbfgs_his = S_LBFGS.optimize()
graphing.plot_w_norm(w_lbfgs_his)
graphing.plot_f_norm(f_lbfgs_his)

