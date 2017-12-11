"""
Linearly convergent stochastic quasi newton
"""
import numpy as np
import collections


class StochasticLBFGS(object):
    def __init__(self, obj, m=None, M=10, L=5, b=20, bH=200, eta=0.05):
        self.n = obj.n # Dimension of solution
        self.N = obj.N # Number of sub functions
        self.b = b # Number of samples to estimate gradient
        self.bH = bH # Number of samples to estimate hessian
        self.eta = eta # Constant step size
        self.obj = obj
        if m is None: # If no specified, m gets N/b
            self.m = int(self.N/b) # Number of sub-steps in each iteration
        else:
            self.m = int(m)

        self.M = M # Maximum number of s,y pairs stored in memory
        self.L = L # Interval to update Hr

        self.eps = 1e-6
        self.max_iters = 100
        self.sy = collections.deque(maxlen=M) # This stores the sr yr pairs

    def optimize(self):
        # Init: r = 0, H0 = I
        r = 0
        # Starting w_0
        w0 =  np.random.random((self.n,))
        w_ = [w0] # Keep a history of the iterates
        f_ = [self.obj.evaluate(w0)] # Keep a history of the f_values

        # The inner loop only requires u_r and u_r-1, so keep only two u variables
        u_prev = np.zeros((self.n,))

        for k in range(self.max_iters):  # For k = 0, ...
            uk = self.obj.compute_full_gradient(w_[k])  # Compute full gradient with latest w
            x_ = [w_[k]]  # x_0 = w_k

            for t in range(self.m): # For t = 0,...,m-1
                s = np.random.choice(self.N, self.b) # Sample b sub functions to estimate gradient
                # Compute stochastic gradient
                grad_s_x = self.compute_stochastic_gradient(x_[t], s) # dF(xt)
                grad_s_w = self.compute_stochastic_gradient(w_[k], s) # dF(wk)

                vt = grad_s_x - grad_s_w + uk # Compute reduced variance gradient vt
                Hvt = self.compute_H_vt(vt)
                x_next = x_[t] - self.eta*Hvt # x_t+1 = x_t - eta Hrvt
                x_.append(x_next) # Update x

                if t % self.L == 0 and t != 0: # Update Hr (by adding a new pair of sr, yr)
                    r += 1
                    # Compute ur
                    ur = self.compute_ur(x_, t)
                    # Randomly choose bH samples to estimate hessian
                    sH = np.random.choice(self.N, self.bH)

                    # Compute stochastic hessian
                    sr = ur - u_prev
                    stochastic_hessian = self.compute_stochastic_hessian(ur, sH)
                    yr = np.dot(stochastic_hessian, sr)

                    # Add the new sr, yr, since maxLen has been specified, this will automatically
                    # Remove the extra elements
                    self.sy.append((sr,yr))
                    u_prev = ur

            # w_k+1 chosen randomly from the set of m values of x or the last x
            w_next = x_[-1]
            f_.append(self.obj.evaluate(w_next))
            w_.append(w_next)

        return w_, f_

    def compute_ur(self, x_, t):
        ur = np.zeros((self.n, ))
        for i in range(max(t-self.L,0), t+1):
            ur += x_[i]
        return ur/self.L

    def compute_stochastic_hessian(self, x, sH):
        hess = np.zeros((self.n, self.n))
        for i in sH:
            hess += self.obj.compute_sub_hessian(x, i)
        return hess/self.bH

    def compute_stochastic_gradient(self, x, s):
        grad = np.zeros_like(x)
        for i in s:
            grad += self.obj.compute_sub_gradient(x,i)
        return grad/self.b

    def compute_H_vt(self, vt):
        q = vt # Suppose dF is ones -> the final answer will give Hk
        if len(self.sy) < self.m: # If we don't have enough s,y pairs, use gradient descent
            return vt

        alpha = []
        (s,y) = self.sy.pop() # Get s_k-1 and y_k-1 to compute gammak
        self.sy.append((s,y)) # Add back because we will need it again
        gamma = np.dot(s.T,y)/np.dot(y.T,y)

        # Go in reverse
        for (s,y) in reversed(self.sy):
            rho = 1/np.dot(s.T, y)
            a = rho*(np.dot(s.T, q))
            q = q - a*y
            alpha.append(a)

        H = gamma*np.eye(self.n)
        r = np.dot(H, q)
        # Go forward
        i = self.m-1
        for (s,y) in self.sy:
            beta = rho*np.dot(y.T,r)
            r += s*(alpha[i]-beta)
            i -= 1
        return r # This is Hk vt

    # Update Hr based on the two loops formula
    # Reference to Nocedal_Wright
    def update_H(self):
        q = np.ones((self.n,)) # Suppose dF is ones -> the final answer will give Hk
        alpha = []
        (s,y) = self.sy.pop() # Get s_k-1 and y_k-1 to compute H_k0
        self.sy.append((s,y))
        gamma = np.dot(s.T,y)/np.dot(y.T,y)

        # Go in reverse
        for (s,y) in reversed(self.sy):
            rho = 1/np.dot(s.T,y)
            a = rho*(np.dot(s.T,q))
            q -= a*y
            alpha.append(a)

        H = gamma*np.eye(self.n)
        r = np.dot(H, q)

        # Go forward
        alpha.reverse() # First reverse alpha to make it consisten with the two loops formulation
        for i, (s,y) in enumerate(self.sy):
            beta = rho*np.dot(y.T,r)
            r += s*(alpha[i]-beta)

        return r # This is Hk




