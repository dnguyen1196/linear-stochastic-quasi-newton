# Stochastic Quasi Newton method
- Explore and implement the stochastic quasi newton method presented in paper
- Implement the heuristic method to increment L every Z iteration (thereby adding
a hyperparameter Z)
- (Extend and implement the idea from the paper on linear convergence, limited memory
SQN)


# Objective functions
Look at >= 2 different objective functions to see how
the different algorithm perform differently

How initializations affect the algorithm



# A new algorithm inspired by SAGA?
    Init: r = 0, H0 = I
    
    for k = 1 : inf        
        Pick i at random
        x_k+1 = wk
        Store dfi(x_k+1) or all j in the table
        phi_k+1 = wk - gamma[dfi(x_k+1)-dfi(x_k)+1/n average table]
            
        if t % L == 0
            Update Hr 

So the table contains some sort of values of all the gradient
at x_k though it doesnt have to store x_k explicitly, just
the gradient            
