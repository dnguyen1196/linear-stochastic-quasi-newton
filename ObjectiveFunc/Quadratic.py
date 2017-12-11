"""
This quadratic function is sum(1/2x'Aix-bi'x)
"""
import numpy as np
from numpy import dot


class QuadraticFunction(object):
    def __init__(self, N, n):
        np.random.seed(seed=317) # For testing purposes and consistency

        self.N = N
        self.n = n
        self.A_ = []
        self.b_ = []
        for i in range(N):
            Ai = np.random.rand(n,n)
            bi = np.random.rand(n,)
            self.A_.append(dot(Ai.T, Ai))
            self.b_.append(bi)

    def compute_full_gradient(self, w):
        grad = np.zeros((self.n,))
        for i in range(self.N):
            grad += self.compute_sub_gradient(w, i)
        return grad

    def compute_sub_gradient(self, w, i):
        return (1.0/self.N)*(dot(self.A_[i], w)-self.b_[i])

    def compute_sub_hessian(self, w, i):
        return 1.0/self.N*self.A_[i]

    def evaluate(self, w):
        f = 0.0
        for i in range(len(self.A_)):
            f += 1/2*dot(w.T, dot(self.A_[i], w))-dot(self.b_[i], w)
        return f/self.N