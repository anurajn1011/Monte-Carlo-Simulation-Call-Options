#this file will contain two functions to compute stochastic events

#debug imports, clean it up afterwards
import sys
import os
sys.path.append(r"C:\Users\User\Documents\Projects\Monte-Carlo-Simulation-Call-Options\optimization")

#imports
import scipy.stats as stats
import numpy as np
from numpy.linalg import cholesky
from QuadraticResampling import QuadraticResampling

#cholesky decomposition for dW vector
def correlatedWienerProcess(delta_T, num_steps, num_traj, bool_interest=False, bool_volatility=False, correlation_coefficient_S_rho=None, correlation_coefficient_S_sigma=None):
    if not bool_interest and not bool_volatility:
        print("Error: Not computing anything")
    #we are only computing either stochastic interest or stochastic volatility
    if bool_interest ^ bool_volatility:
        #computing interest
        if bool_interest:
            matrix = np.array([[delta_T, correlation_coefficient_S_rho * delta_T], [correlation_coefficient_S_rho * delta_T, delta_T]])
            lower_triangular = cholesky(matrix)
            for _ in range(num_steps):
                delZ = np.random.randn(2, num_traj)
                delZ = QuadraticResampling(delZ, np.zeros((2, 1)), np.identity(2))
                delW = lower_triangular @ delZ
                return delW
        else:
            print("Stochastic volatility has not yet been implemented: ", correlation_coefficient_S_sigma)

#Cox-Ingersoll-Ross model for interest rate
def CoxIngersollRoss(kappa, theta, sigma, initial_interest, correlation_coefficient, initial_asset_price, exercise_price, time_to_maturity, num_steps, num_trajectories):
    '''
    Compute the covariance matrix of two associated Wiener processes of the asset and the interest rate. Performing the Cholesky decomposition,
    we are then able to transform a normally distributed vector into one that has the desired correlation, as provided by L in the CD. This allows us
    to compute the dW vector (dW_S, dW_r), as dW = LdZ where dZ ~ N(0, I^2). Subsequently, compute CIR.

    kappa - speed of adjustment to mean, theta
    theta - mean
    sigma - volatility
    '''

    #computing dW vector
    dW = correlatedWienerProcess(delta_T=time_to_maturity / num_steps, num_steps=num_steps, num_traj=num_trajectories, bool_interest=True, correlation_coefficient=correlation_coefficient)

print(correlatedWienerProcess(1/100, 100, 5000, bool_interest=True, correlation_coefficient_S_rho=-0.2))