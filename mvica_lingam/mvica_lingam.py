"""
Python implementation of the Multiview ICA-based LiNGAM algorithm.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from multiviewica import multiviewica
from shica import shica_j, shica_ml


def mvica_lingam(
    X,
    ica_algo="shica_ml",
    max_iter=3000,
    tol=1e-8,
    random_state=None,
):
    """Implementation of ICA-based multiview LiNGAM model.

    Parameters
    ----------
    X : ndarray, shape (n_views, n_components, n_samples)
        Training data, where ``n_views`` is the number of views, 
        ``n_components`` is the number of components, and ``n_samples`` is
        the number of samples.
    
    ica_algo : string, optional (default="shica_ml")
        The multiview ICA algorithm used in the first step.
        It can be either ``shica_ml``, ``shica_j``, or ``multiviewica``.
        Here are the default parameters of:
            ``shica_ml``: max_iter=3000; tol=1e-8
            ``shica_j``: max_iter=10000; tol=1e-5
            ``multiviewica``: max_iter=1000; tol=1e-3.
    
    max_iter : int, optional (default=3000)
        The maximum number of iterations of the multiview ICA algorithm.
    
    tol : float, optional (default=1e-8)
        The tolerance parameter of the multiview ICA algorithm.
    
    random_state : int, optional (default=None)
        ``random_state`` is the seed used by the random number generator.
    
    Returns
    -------
    P : ndarray, shape (n_components, n_components)
        Causal order represented by a permutation matrix.
    
    B : ndarray, shape (n_views, n_components, n_components)
        Causal effects represented by matrices B[i] as close as possible to
        strictly lower triangular.

    Sigmas: ndarray, shape (n_views, n_components)
        Noise covariances.

    S_avg: ndarray, shape (n_components, n_samples)
        Source estimates.
    """
    m, p, n = X.shape
    
    # Step 1: use a multiview ICA algorithm
    if ica_algo == "shica_ml":
        W, Sigmas, S_avg = shica_ml(X, max_iter=max_iter, tol=tol)
    elif ica_algo == "shica_j":
        W, Sigmas, S_avg = shica_j(X, max_iter=max_iter, tol=tol)
    elif ica_algo == "multiviewica":
        _, W, S_avg = multiviewica(
            X, max_iter=max_iter, tol=tol, random_state=random_state)
        Sigmas = np.ones((m, p))
    else:
        raise ValueError(
            "ica_algo should be either 'shica_ml', 'shica_j', or 'multiviewica'")
    
    # Step 2: find permutation Q
    row_index, col_index = linear_sum_assignment(np.sum(1 / np.abs(W), axis=0))
    QW = ...

    # Step 3: scaling
    D = np.array([np.diag(QW[i]) for i in range(m)])[:, :, np.newaxis]  # shape (m, p, 1)
    DQW = QW / D

    # Step 4: causal effects
    B = np.array([np.eye(p)] * m) - DQW

    # Step 5: estimate the causal order
    P = ...
    
    return P, B, Sigmas, S_avg
