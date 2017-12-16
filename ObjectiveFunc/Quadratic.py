"""
This quadratic function is sum(1/2x'Aix-bi'x)
"""
import numpy as np
from numpy import dot
from numpy.linalg import svd


class QuadraticFunction(object):
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.A_ = []
        self.b_ = []
        # Generate well conditioned matrix
        self.generate_matrices()

    def generate_matrices(self):
        for i in range(self.N):
            A = np.random.rand(self.n, self.n)
            U, S, V = svd(A)
            eigs = np.diag(np.linspace(1,2,self.n))
            A = dot(U, dot(eigs, U.T))
            bi = np.random.rand(self.n,)
            self.A_.append(A)
            self.b_.append(bi)

    def compute_full_gradient(self, w):
        grad = np.zeros((self.n,))
        for i in range(self.N):
            grad += self.compute_sub_gradient(w, i)
        return grad/self.N

    def compute_sub_gradient(self, w, i):
        return dot(self.A_[i], w)-self.b_[i]

    def compute_sub_hessian(self, w, i):
        return self.A_[i]

    def evaluate(self, w):
        f = 0.0
        for i in range(len(self.A_)):
            f += 1/2*dot(w.T, dot(self.A_[i], w))-dot(self.b_[i], w)
        return f/self.N