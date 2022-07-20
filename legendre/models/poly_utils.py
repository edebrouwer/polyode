import numpy as np

def get_mu_covar(cns):
    mu_cns = cns.mean(1)
    cov_cns = np.cov(cns)
    return mu_cns, cov_cns

def get_cns(X,y, degree):
    """
    Compute the coefficients of the Legendre polynomials approximating y on X up to degree degree
    """
    cns = np.polynomial.legendre.legfit(X, y, deg = degree)
    return cns

