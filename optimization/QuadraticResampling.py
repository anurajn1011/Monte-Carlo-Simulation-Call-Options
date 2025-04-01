#quadratic resampling
#method to achieve higher agreement between sample and theoretical params

import numpy as np
from numpy.linalg import cholesky, inv

def QuadraticResampling(matrix, theoretical_mean, theoretical_covariance):
    '''
        Parameters:
            matrix - Matrix of a simulation
            theoretical_mean - Theoretical mean vector
            theoretical_covariance - Theoretical covariance vector
    '''
    num_trajectories = len(matrix[0])
    #sample parameters
    sample_mean = np.mean(matrix, axis=1, keepdims=True)
    sample_mean_tiled = np.tile(sample_mean, (1, num_trajectories))
    print("Num of cols: ", len(matrix[0]))
    print("matrix shape: ", matrix.shape)
    print("repmat shape: ", np.tile(sample_mean, (2, num_trajectories)).shape)
    sample_covariance = np.cov(np.transpose(matrix - sample_mean_tiled), rowvar=False)
    # sample_covariance = np.cov(np.transpose(matrix - np.tile(sample_mean, (1, num_trajectories))), rowvar=False)

    #cholesky decomposition
    lower_sample_covariance = np.transpose(cholesky(sample_covariance))
    lower_theoretical_covariance = np.transpose(cholesky(theoretical_covariance))

    return lower_theoretical_covariance @ inv(lower_sample_covariance) @ (matrix - np.tile(sample_mean, (1, num_trajectories))) + np.tile(theoretical_mean, (1, num_trajectories))