import numpy as np
from scipy.stats import gennorm, pearsonr
from picard import amari_distance
import lingam
from micado.micado import micado


# function that samples data according to our model
# we use similar parameters as in Fig. 2 of the ShICA paper
def sample_data(
    m,
    p,
    n,
    noise_level=1.,
    density="gauss_super",
    beta1=1,
    beta2=3,
    nb_gaussian_disturbances=0,
    nb_equal_variances=0,
    random_state=None,
    shared_causal_ordering=True,
):
    rng = np.random.RandomState(random_state)
    if density == "gauss_super":
        # sources
        S_ng = rng.laplace(size=(p-nb_gaussian_disturbances, n))
        S_g = rng.normal(size=(nb_gaussian_disturbances, n))
        S = np.vstack((S_ng, S_g))
        # noise variances
        sigmas = np.ones((m, p)) * 1 / 2
        if nb_gaussian_disturbances != 0:
            sigmas[:, -nb_gaussian_disturbances:] = rng.uniform(size=(m, nb_gaussian_disturbances))
            if nb_equal_variances > 0:
                indices = rng.choice(m, size=nb_equal_variances, replace=False)
                sigmas[indices, -nb_gaussian_disturbances:] = sigmas[indices, -nb_gaussian_disturbances][:, np.newaxis]
    elif density == "sub_gauss_super":
        # sources
        S1 = gennorm.rvs(beta1, size=(p//3, n), random_state=random_state)
        S2 = gennorm.rvs(2, size=(p-2*(p//3), n), random_state=random_state)  # Gaussian
        S3 = gennorm.rvs(beta2, size=(p//3, n), random_state=random_state)
        S = np.vstack((S1, S2, S3))
        # noise variances
        sigmas = rng.uniform(size=(m, p))
    else:
        raise ValueError("The parameter 'density' should be either 'gauss_super' or 'sub_gauss_super'")
    
    # noise
    N = noise_level * rng.normal(scale=sigmas[:, :, np.newaxis], size=(m, p, n))
    # causal effect matrices T
    T = rng.normal(size=(m, p, p))
    for i in range(m):
        T[i][np.triu_indices(p, k=0)] = 0  # set the strictly upper triangular part to 0
    # causal order
    if shared_causal_ordering:
        P = np.eye(p)[rng.permutation(p)]
    else:
        P = np.array([np.eye(p)[rng.permutation(p)] for _ in range(m)])
    # causal effect matrices B
    if shared_causal_ordering:
        B = P.T @ T @ P
    else:
        B = np.array([Pi.T @ Ti @ Pi for Pi, Ti in zip(P, T)])
    # mixing matrices
    A = np.linalg.inv(np.eye(p) - B)
    # observations
    X = np.array([Ai @ Si for Ai, Si in zip(A, S + N)])
    return X, B, T, P, A


def run_experiment(
    m,
    p,
    n,
    noise_level=1.,
    density="gauss_super",
    beta1=1,
    beta2=3,
    nb_gaussian_disturbances=0,
    nb_equal_variances=0,
    ica_algo="shica_ml",
    random_state=None,
    shared_causal_ordering=True,
    new_find_order_function=False,
):
    if density == "sub_gauss_super":
        nb_gaussian_disturbances = p - 2 * (p // 3)
    # generate observations X, causal order(s) P, and causal effects B and T
    X, B, T, P, A = sample_data(
        m=m,
        p=p,
        n=n,
        noise_level=noise_level,
        density=density,
        beta1=beta1,
        beta2=beta2,
        nb_gaussian_disturbances=nb_gaussian_disturbances,
        nb_equal_variances=nb_equal_variances,
        random_state=random_state,
        shared_causal_ordering=shared_causal_ordering,
    )

    # apply either our method, Multi Group DirectLiNGAM, or LiNGAM
    if ica_algo in ["multiviewica", "shica_j", "shica_ml"]:
        # apply our main function to retrieve B, T, P, and W;
        B_estimates, T_estimates, P_estimate, _, W_estimates = micado(
            X, shared_causal_ordering=shared_causal_ordering, ica_algo=ica_algo,
            random_state=random_state, new_find_order_function=new_find_order_function,
            return_full=True)
        if not shared_causal_ordering:
            P_estimates = P_estimate  # shape (m, p, p)
    elif ica_algo == "multi_group_direct_lingam":
        # apply Multi Group DirectLiNGAM to retrieve B, T, P, and W
        model = lingam.MultiGroupDirectLiNGAM()
        model.fit(list(np.swapaxes(X, 1, 2)))
        # causal order P
        P_estimate = np.eye(p)[model.causal_order_]
        # causal effect matrices B and T
        B_estimates = np.array(model.adjacency_matrices_)
        T_estimates = P_estimate @ B_estimates @ P_estimate.T
        # reconstruct what would be unmixing matrices W
        W_estimates = np.eye(p) - B_estimates
    elif ica_algo == "lingam":
        # apply LiNGAM to retrieve B, T, P, and W
        B_estimates = []
        T_estimates = []
        P_estimates = []
        model = lingam.ICALiNGAM()
        for i in range(m):
            model.fit(np.swapaxes(X[i], 0, 1))
            # causal order P
            P_estimate = np.eye(p)[model.causal_order_]
            P_estimates.append(P_estimate)
            # causal effect matrices B and T
            B_estimate = np.array(model._adjacency_matrix)
            B_estimates.append(B_estimate)
            T_estimate = P_estimate @ B_estimate @ P_estimate.T
            T_estimates.append(T_estimate)
        B_estimates = np.array(B_estimates)
        T_estimates = np.array(T_estimates)
        P_estimates = np.array(P_estimates)  # shape (m, p, p) and not (p, p)
        # reconstruct unmixing matrices W
        W_estimates = np.eye(p) - B_estimates
    else:
        raise ValueError("Wrong ica_algo.")
    
    # errors
    def compute_error_P(P1, P2, method="exact"):
        if method == "exact":
            error_P = 1 - (P1 == P2).all()
            return error_P
        else:
            corr = pearsonr(np.argmax(P1, axis=1), np.argmax(P2, axis=1))[0]
            return corr

    if shared_causal_ordering:
        # P has shape (p, p)
        if ica_algo == "lingam":
            # P_estimates has shape (m, p, p)
            # error_P = np.mean([1 - (Pe == P).all() for Pe in P_estimates])
            error_P_spearmanr = np.mean(
                [compute_error_P(Pe, P, method="spearmanr") 
                 for Pe in P_estimates])
            error_P_exact = np.mean(
                [compute_error_P(Pe, P, method="exact") 
                 for Pe in P_estimates])
        else:
            # P_estimate has shape (p, p)
            # error_P = 1 - (P_estimate == P).all()
            error_P_spearmanr = compute_error_P(P_estimate, P, method="spearmanr")
            error_P_exact = compute_error_P(P_estimate, P, method="exact")
    else:
        # P has shape (m, p, p)
        if ica_algo == "multi_group_direct_lingam":
            # P_estimate has shape (p, p)
            # error_P = np.mean([1 - (P_estimate == Pi).all() for Pi in P])
            error_P_spearmanr = np.mean(
                [compute_error_P(P_estimate, Pi, method="spearmanr") for Pi in P])
            error_P_exact = np.mean(
                [compute_error_P(P_estimate, Pi, method="exact") for Pi in P])
        else:
            # P_estimates has shape (m, p, p)
            # error_P = np.mean([1 - (Pe == Pi).all() for Pe, Pi in zip(P_estimates, P)])
            error_P_spearmanr = np.mean(
                [compute_error_P(Pe, Pi, method="spearmanr")
                 for Pe, Pi in zip(P_estimates, P)])
            error_P_exact = np.mean(
                [compute_error_P(Pe, Pi, method="exact")
                 for Pe, Pi in zip(P_estimates, P)])
    error_B = np.mean((B_estimates - B) ** 2)
    error_B_abs = np.mean(np.abs(B_estimates - B))
    error_T = np.mean((T_estimates - T) ** 2)
    error_T_abs = np.mean(np.abs(T_estimates - T))
    amari = np.mean([amari_distance(Wi, Ai) for Wi, Ai in zip(W_estimates, A)])
    
    # output
    output = {
        "m": m,
        "p": p,
        "n": n,
        "noise_level": noise_level,
        "ica_algo": ica_algo,
        "nb_gaussian_disturbances": nb_gaussian_disturbances,
        "nb_equal_variances": nb_equal_variances,
        "random_state": random_state,
        "error_B": error_B,
        "error_B_abs": error_B_abs,
        "error_T": error_T,
        "error_T_abs": error_T_abs,
        "error_P_spearmanr": error_P_spearmanr,
        "error_P_exact": error_P_exact,
        "amari_distance": amari,
    }
    return output
