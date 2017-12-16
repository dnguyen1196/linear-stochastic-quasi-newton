"""
Linearly convergent stochastic quasi newton
"""
import numpy as np
import collections

class SVRG_LBFGS(object):
    def __init__(self, obj,w0=None, m=10, M=10, L=5, b=20, bH=200, eta=0.5):
        self.n = obj.n # Dimension of solution
        self.N = obj.N # Number of sub functions
        self.b = b # Number of samples to estimate gradient
        self.bH = bH # Number of samples to estimate hessian
        self.eta = eta # Constant step size
        self.obj = obj
        self.m = m

        self.M = M # Maximum number of s,y pairs stored in memory
        self.L = L # Interval to update Hr
        self.w0 = w0
        self.max_iters = 100
        self.sy = collections.deque(maxlen=M) # This stores the sr yr pairs
        self.eps = 1e-9

    def optimize(self):
        # Init: H0 = I
        # Starting w_0
        if self.w0 is None:
            w_ = [np.random.random((self.n,))] # Random starting point
        else:
            w_ = [self.w0]

        f_ = [self.obj.evaluate(w_[0])] # Keep a history of the f_values
        # The inner loop only requires u_r and u_r-1, so keep only two u variables
        u_prev = np.zeros((self.n,))

        for k in range(self.max_iters):  # For k = 0, ...
            grad_k = self.obj.compute_full_gradient(w_[k])  # Compute full gradient with latest w
            x_ = [w_[k]]  # x_0 = w_k

            for t in range(self.m): # For t = 0,...,m-1
                s = np.random.choice(self.N, self.b) # Sample b sub functions to estimate gradient
                # Compute stochastic gradient
                grad_s_x = self.compute_stochastic_gradient(x_[t], s) # dF(xt)
                grad_s_w = self.compute_stochastic_gradient(w_[k], s) # dF(wk)

                vt = grad_s_x - grad_s_w + grad_k # Compute reduced variance gradient vt

                Hvt = self.compute_H_vt(vt) # Compute step direction
                x_next = x_[t] - self.eta*Hvt # x_t+1 = x_t - eta Hrvt
                x_.append(x_next) # Update x

                if t % self.L == 0 and t != 0: # Update Hr (by adding a new pair of sr, yr)
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
            w_next = x_[np.random.choice(self.m)+1]

            if np.linalg.norm(w_next-w_[k]) < self.eps: # Stopping condition
                break

            f_.append(self.obj.evaluate(w_next))
            w_.append(w_next)

        return w_, f_

    def compute_ur(self, x_, t):
        ur = np.zeros((self.n, ))
        count = 0
        for i in range(max(t-self.L,0), t+1):
            count += 1
            ur += x_[i]
        return ur/count

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
        q = vt
        if len(self.sy) == 0:
            # If we don't have enough s,y pairs, use gradient descent
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
        i = len(alpha)-1
        for (s,y) in self.sy:
            beta = rho*np.dot(y.T,r)
            r += s*(alpha[i]-beta)
            i -= 1

        return r # This is Hk vt
