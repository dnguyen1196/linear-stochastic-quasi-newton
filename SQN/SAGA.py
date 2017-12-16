import numpy as np


class SAGA(object):
    def __init__(self, obj, eta=0.01, w0=None):
        self.obj = obj
        self.N = obj.N
        self.n = obj.n
        self.maxiters = 300
        self.eta = eta
        self.w0 = w0
        self.eps = 1e-12

    def optimize(self):
        if self.w0 is None:
            w_ = [np.random.random((self.n,))] # Random starting point
        else:
            w_ = [self.w0]

        f_ = [self.obj.evaluate(w_[0])] # Keep a history of the f_values

        phi_array = self.initialize_di(w_[0])
        for k in range(self.maxiters):
            ik = np.random.choice(self.N)
            df_ik = self.obj.compute_sub_gradient(w_[k], ik)
            w_next = w_[k] - self.eta * (np.mean(phi_array, axis=0) + df_ik - phi_array[ik, :])
            phi_array[ik, :] = df_ik

            if np.linalg.norm(w_next-w_[k]) < self.eps or np.any(np.isnan(w_next)): # Stopping condition
                break
            w_.append(w_next)
            f_.append(self.obj.evaluate(w_next))

        return w_, f_

    def initialize_di(self, w):
        d_array = np.zeros((self.N, self.n))
        for i in range(self.N):
            d_array[i, :] = self.obj.compute_sub_gradient(w, i)
        return d_array