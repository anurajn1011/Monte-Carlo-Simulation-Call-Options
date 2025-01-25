import math
import numpy as np
import scipy.stats as stats

#control variates for variance reduction

def monte_carlo_option_price(S0, K, T):
    #S0 - Initial price of asset
    #K - Strike Price
    #T - Time to maturity

    #initial params, for real cases compute using data
    sigma = 0.2
    r = 0.05
    NbTraj = 5000
    NbPas = 100
    DeltaT = T/NbPas
    tau = T

    #price vector
    present = S0 * np.ones([NbTraj, 1])
    next = np.zeros([NbTraj, 1])

    #control variables
    y1 = np.zeros([NbTraj, 1])
    y2 = np.zeros([NbTraj, 1])

    #simulate trajectories
    for i in range(NbPas):
        dw = math.sqrt(DeltaT) * np.random.normal(0, 1, size=(NbTraj, 1))
        if i < NbPas and tau - DeltaT > 0:
            tau -= DeltaT
            d1 = ((np.log(present / K) + (r + (sigma**2 / 2)) * tau) / (sigma * np.sqrt(tau)))
            y1 = y1 + stats.norm.cdf(d1) * present * dw * sigma
            y2 = y2 + (stats.norm.cdf(d1) - 1) * present * dw * sigma
        next = present + (r * present * DeltaT + sigma * present * dw)
        present = next
    
    #Call option
    call = np.exp(-r * T) * np.maximum(0, (present - K))
    expectation_call = np.sum(call) / NbTraj
    expectation_control_1 = 0

    cov_call = ((call - expectation_call) * (y1 - expectation_control_1)) / (NbTraj - 1)
    var_call = ((y1 - expectation_control_1) * (y1 - expectation_control_1)) / (NbTraj - 1)
    #optimal alpha
    alpha1 = -cov_call / var_call

    PrixControl1 = call + alpha1 * (y1 - expectation_control_1)
    call_price = np.sum(PrixControl1) / NbTraj

    #Put Option
    put = np.exp(-r * T) * np.maximum(0, K - present)
    expectation_put = np.sum(put) / NbTraj
    expectation_control_2 = 0
    cov_put = ((put - expectation_put) * (y2 - expectation_control_2)) / (NbTraj - 1)
    var_put = (((y2 - expectation_control_2)) * ((y2 - expectation_control_2))) / (NbTraj - 1)
    #optimal alpha
    alpha2 = -cov_put / var_put

    PrixControl2 = put + alpha2 * (y2 - expectation_control_2)
    put_price = np.sum(PrixControl2) / NbTraj


    return (call_price, put_price)