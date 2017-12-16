import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from ObjectiveFunc.Quadratic import QuadraticFunction
from ObjectiveFunc.Logistic import LogisticRegression
from SQN.SVRG_SQN import SVRG_LBFGS
from SQN.SAGA import SAGA
from SQN.SVRG import SVRG_GD
from SQN.SAGA_SQN import SAGA_LBFGS
from Visualization import Grapher as graphing

N = 400
n = 5

np.random.seed(seed=317)
# For testing and consistency check

quadratic_cost = QuadraticFunction(N=N, n=n)
logistic_cost = LogisticRegression(N=N, n=n)

# w0 = np.ones((n,))
# w0 = np.random.random((n,))
w0 = np.ones((n,))/2
# Quadratic/ strongly convex function
# SVRG variants
VR_LBFGS_quadratic = SVRG_LBFGS(obj=quadratic_cost, w0=w0, eta=0.5)
SVRG_quadratic = SVRG_GD(obj=quadratic_cost, w0=w0, eta=0.5)

# SAGA variants
# SAGA_GD = SAGA(obj=quadratic_cost,w0=w0, eta=0.01)
# SAGA_BFGS = SAGA_LBFGS(obj=quadratic_cost,w0=w0, eta=0.01)

VR_LBFGS_logistic = SVRG_LBFGS(obj=logistic_cost, w0=w0, eta=0.1)
SVRG_logistic = SVRG_GD(obj=logistic_cost, w0=w0, eta=0.05)

"""
Do optimization
"""
# w_lbfgs, f_lbfgs = VR_LBFGS.optimize()
# w_svrg, f_svrg = SVRG.optimize()

w_lbfgs_log, f_lbfgs_log = VR_LBFGS_logistic.optimize()
# w_svrg_log, f_svrg_log = SVRG_logistic.optimize()

print(f_lbfgs_log)

# w_saga, f_saga = SAGA_GD.optimize()
# w_saga_bfgs, f_saga_bfgs = SAGA_BFGS.optimize()

"""
Do plotting
"""
# graphing.plot_w_norm(w_lbfgs)
# graphing.plot_f_norm(f_svrg_log)
# graphing.plot_compare_log_f_norm(w_svrg, w_lbfgs, "svrg", "svrg lbfgs")
# graphing.plot_compare_log_f_norm(f_saga, f_saga_bfgs, "saga", "saga lbfgs")
# graphing.plot_compare_log_f_norm(w_svrg_log, w_lbfgs_log, 'svrg', 'l-bfgs')