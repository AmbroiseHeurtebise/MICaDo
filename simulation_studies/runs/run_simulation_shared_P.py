import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from picard import amari_distance
import lingam
from mvica_lingam.mvica_lingam import mvica_lingam


# function that samples data according to our model
# we use similar parameters as in Fig. 2 of the ShICA paper
def sample_data(m, p, n, nb_gaussian_sources=0, rng=None):
    # sources
    S_ng = rng.laplace(size=(p-nb_gaussian_sources, n))
    S_g = rng.normal(size=(nb_gaussian_sources, n))
    S = np.vstack((S_ng, S_g))
    # noise
    sigmas = np.ones((m, p)) * 1 / 2
    if nb_gaussian_sources != 0:
        sigmas[:, -nb_gaussian_sources:] = rng.uniform(size=(m, nb_gaussian_sources))
    N = rng.normal(scale=sigmas[:, :, np.newaxis], size=(m, p, n))
    # causal effect matrices T
    T = rng.normal(size=(m, p, p))
    for i in range(m):
        T[i][np.triu_indices(p, k=0)] = 0  # set the strictly upper triangular part to 0
    # causal order
    P = np.eye(p)
    rng.shuffle(P)
    # causal effect matrices B
    B = P.T @ T @ P
    # mixing matrices
    A = np.linalg.inv(np.eye(p) - B)
    # observations
    X = np.array([Ai @ Si for Ai, Si in zip(A, S + N)])
    return X, B, T, P, A


def run_experiment(m, p, n, nb_gaussian_sources, random_state, ica_algo):
    rng = np.random.RandomState(random_state)
    # generate observations X, causal order P, and causal effects B and T
    X, B, T, P, A = sample_data(m, p, n, nb_gaussian_sources, rng)
    # apply either our method, Multi Group DirectLiNGAM, or LiNGAM
    if ica_algo in ["multiviewica", "shica_j", "shica_ml"]:
        # apply our main function to retrieve B, T, P, and W
        B_estimates, T_estimates, P_estimate, _, W_estimates = mvica_lingam(
            X, ica_algo=ica_algo, random_state=random_state)
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
    if ica_algo != "lingam":
        error_P = 1 - (P_estimate == P).all()
    else:
        error_P = np.mean([1 - (Pi == P).all() for Pi in P_estimates])
    error_B = np.mean((B_estimates - B) ** 2)
    error_B_abs = np.mean(np.abs(B_estimates - B))
    error_T = np.mean((T_estimates - T) ** 2)
    error_T_abs = np.mean(np.abs(T_estimates - T))
    amari = np.mean([amari_distance(Wi, Ai) for Wi, Ai in zip(W_estimates, A)])
    
    # output
    output = {
        "ica_algo": ica_algo,
        "nb_gaussian_sources": nb_gaussian_sources,
        "n": n,
        "random_state": random_state,
        "error_B": error_B,
        "error_B_abs": error_B_abs,
        "error_T": error_T,
        "error_T_abs": error_T_abs,
        "error_P": error_P,
        "amari_distance": amari,
    }
    return output


# fixed parameters
m = 5
p = 4
N_JOBS = 10

# varying parameters
nb_gaussian_sources_list = [0, 2, 4]
nb_seeds = 50
random_state_list = np.arange(nb_seeds)
n_list = np.logspace(2, 4, 21, dtype=int)
algo_list = ["multiviewica", "shica_j", "shica_ml", "multi_group_direct_lingam", "lingam"]

# run experiment
nb_expes = len(nb_gaussian_sources_list) * len(random_state_list) * len(n_list) * len(algo_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        nb_gaussian_sources=nb_gaussian_sources,
        random_state=random_state,
        ica_algo=ica_algo,
    ) for n, nb_gaussian_sources, random_state, ica_algo
    in product(n_list, nb_gaussian_sources_list, random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/mvica_lingam/simulation_studies/results/shared_P/"
save_name = f"DataFrame_with_{nb_seeds}_seeds_and_4_metrics"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
