import numpy as np
from numpy import dot
from numpy import outer
from numpy.linalg import svd

class LogisticRegression():
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.w = np.random.rand(n,)
        self.X = np.random.rand(N,n)

        # A = np.random.rand(self.n, self.n)
        # U, S, V = svd(A)
        # eigs = np.diag(np.linspace(1, 2, self.n))
        # sel. = dot(U, dot(eigs, U.T))

        z = 1/(1+np.exp(-dot(self.X, self.w)))
        self.Z = np.zeros((N,))
        for i in range(self.N):
            if z[i] > 1/2:
                self.Z[i] = 1

    def compute_sub_gradient(self, w, i):
        # Do some computation here
        prod = np.inner(self.X[i, :], w)
        return (1/(1+np.exp(-prod)) - self.Z[i])*self.X[i,:].T

    def compute_full_gradient(self, w):
        grad = np.zeros((self.n,))
        for i in range(self.N):
            grad += self.compute_sub_gradient(w,i)
        return grad

    def evaluate(self, w):
        f = 0.0
        for i in range(self.N):
            prod = np.inner(self.X[i, :],w)
            if self.Z[i] == 0:
                f -= np.log(1-1/(1+np.exp(prod)))
            else:
                f -= np.log(1/(1+np.exp(prod)))
        return f/self.N

    # Function to compute sub_hessian, note that in actual large scale system
    # We do not form hessian explicitly but compute hessian s directly
    def compute_sub_hessian(self, w, i):
        xi = self.X[i, :]
        c = 1/(1+np.exp(-dot(xi,w)))
        return c*(1-c)*outer(xi, xi)