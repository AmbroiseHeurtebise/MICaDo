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
    W_inv = 1 / np.sum([np.abs(Wi.T) for Wi in W], axis=0)  # shape (p, p)
    _, index = linear_sum_assignment(W_inv)
    QW = np.array([Wi[index] for Wi in W])

    # Step 3: scaling
    D = np.array([np.diag(Wi) for Wi in QW])[:, :, np.newaxis]  # shape (m, p, 1)
    DQW = QW / D

    # Step 4: causal effects
    B_hat = np.array([np.eye(p)] * m) - DQW  # B_hat is not yet lower triangular

    # Step 5: estimate the causal order
    order = find_order(B_hat)
    P = np.eye(p)[order]
    B = P @ B_hat @ P.T
    
    return P, B, Sigmas, S_avg


def find_order(B_hat):
    """This function finds a permutation P such that P @ B_hat @ P.T
    is as close as possible to strictly lower triangular.

    Args:
        B_hat : ndarray, shape (m, p, p)
            Causal effect matrices, whose rows and columns are permuted
            by a common permutation P. We assume that ``B_hat`` is such that 
            P @ B_hat @ P.T are close to strictly lower triangular.

    Returns:
        order: ndarray, shape (p,)
            ``order`` represents the permutation in P.
    """
    _, p, _ = B_hat.shape
    B_avg = np.mean(np.abs(B_hat), axis=0)
    B_sort = np.sort(B_avg, axis=1)[:, ::-1]
    B_argsort = np.argsort(B_sort, axis=0)
    order = []
    for i in range(p):
        col = B_argsort[:, i]
        available_id = ~np.isin(B_argsort[:, i], order)
        j = 0
        while not available_id[j]:
            j += 1
        order.append(col[j])
    return order
