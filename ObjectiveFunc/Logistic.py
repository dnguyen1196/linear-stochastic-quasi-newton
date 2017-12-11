import numpy as np
from numpy import dot
from numpy import outer

class LogisticRegression():
    def __init__(self, N, n):
        np.random.seed(seed=317) # For testing and consistency check
        self.N = N
        self.n = n
        self.w = np.random.rand(n,)
        self.X = np.random.rand(N,n)
        z = 1/(1+np.exp(-dot(self.X, self.w)))
        self.Z = np.zeros((N,))
        for i in range(self.N):
            if z[i] > 0:
                self.Z[i] = 1

    def compute_sub_gradient(self, w, i):
        # Do some computation here
        return (1/(1+np.exp(-dot(self.X[i,:],w)))-self.Z[i])*self.X[i,:].T

    def compute_full_gradient(self, w):
        grad = np.zeros((self.n,))
        for i in range(self.N):
            grad += self.compute_sub_gradient(w,i)
        return grad

    def evaluate(self, w):
        f = 0.0
        for i in range(self.N):
            if self.Z[i] == 0:
                f += np.log(1-1/(1+np.exp(dot(self.X[i,:],w))))
            else:
                f += np.log(1/(1+np.exp(dot(self.X[i,:],w))))
        return f/self.N

    # Function to compute sub_hessian, note that in actual large scale system
    # We do not form hessian explicitly but compute hessian s directly
    def compute_sub_hessian(self, w, i):
        xi = self.X[i, :]
        c = 1/(1+np.exp(-dot(xi,w)))
        return c*(1-c)*outer(xi, xi)