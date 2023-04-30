import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve


def exponentiated_quadratic(xa, xb, l_kernel):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -l_kernel * cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1), 
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    # Solve
    solved = solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance


def generate_gp_data(n1,n2,ny, l_kernel = 3,domain = [-1,1]):
    # Compute the posterior mean and covariance 
    """
    n1 : number of (x,y) points to condition upon
    n2 : number of x points in the posterior
    ny : number of trajectories to sample from the posterior
    domain : domain of x points
    l_kernel : the scale of  the RBF kernel used to compute the posterior
    """

    # Define the true function that we want to regress on
    f_sin = lambda x: (np.exp(1-0.01*x**3) + 3 * np.cos(x**2)*np.sin(x)).flatten()

    n1 = n1  # Number of points to condition on (training points)
    n2 = n2  # Number of points in posterior (test points)
    ny = ny  # Number of functions that will be sampled from the posterior
    #domain = (-1, 1)

    # Sample observations (X1, y1) on the function
    X1 = np.linspace(domain[0],domain[1]-0.5,n1)[:,None]
    y1 = f_sin(3*X1)
    # Predict points at uniform spacing to capture function
    X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)
    # Compute posterior mean and covariance
    μ2, Σ2 = GP(X1, y1, X2, lambda x,y : exponentiated_quadratic(x,y, l_kernel = l_kernel))
    # Compute the standard deviation at the test points to be plotted
    σ2 = np.sqrt(np.diag(Σ2))

    # Draw some samples of the posterior
    y2 = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=ny)
    return X1, y1, X2, y2, Σ2
