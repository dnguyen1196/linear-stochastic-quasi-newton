import numpy as np


class SVRG_GD(object):
    def __init__(self, obj, w0=None, eta=0.5, b=20, m=5):
        self.obj = obj
        self.n = obj.n
        self.N = obj.N
        self.b = b # Number of samples to compute stochastic gradient
        self.eta = eta
        self.m = m

        self.w0 = w0
        self.maxiters = 100
        self.eps = 1e-9

    def optimize(self):
        if self.w0 is None:
            w_ = [np.random.random((self.n,))] # Random starting point
        else:
            w_ = [self.w0]

        f_ = [self.obj.evaluate(w_[0])] # Keep a history of the f_values

        for k in range(self.maxiters):
            z_ = [w_[k]]
            mu = self.obj.compute_full_gradient(w_[k])

            for j in range(self.m):
                ik = np.random.choice(self.N)  # Sample random index
                z_cur = z_[-1]
                z_next = z_cur - self.eta * (self.obj.compute_sub_gradient(z_cur, ik)- self.obj.compute_sub_gradient(z_[0], ik)+ mu)
                z_.append(z_next)

            w_next = z_[np.random.choice(self.m)+1] # Pick w_next to be the last z computed
            if np.linalg.norm(w_next-w_[k]) < self.eps: # Stopping condition
                break
            f_.append(self.obj.evaluate(w_next))
            w_.append(w_next)

        return w_, f_

    def compute_stochastic_gradient(self, x, s):
        grad = np.zeros_like(x)
        for i in s:
            grad += self.obj.compute_sub_gradient(x,i)
        return grad/self.b

