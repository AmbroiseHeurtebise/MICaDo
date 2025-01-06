import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from picard import amari_distance
import lingam
from mvica_lingam.mvica_lingam import mvica_lingam


# function that samples data according to our model
# we use similar parameters as in Fig. 2 of the ShICA paper
def sample_data(m, p, n, noise_level=1., nb_gaussian_sources=0, rng=None):
    # sources
    S_ng = rng.laplace(size=(p-nb_gaussian_sources, n))
    S_g = rng.normal(size=(nb_gaussian_sources, n))
    S = np.vstack((S_ng, S_g))

    # noise
    sigmas = np.ones((m, p)) * 1 / 2
    if nb_gaussian_sources != 0:
        sigmas[:, -nb_gaussian_sources:] = rng.uniform(size=(m, nb_gaussian_sources))
    N = noise_level * rng.normal(scale=sigmas[:, :, np.newaxis], size=(m, p, n))

    # causal effect matrices
    B = rng.normal(size=(m, p, p))
    for i in range(m):
        B[i][np.triu_indices(p, k=0)] = 0  # set the strictly upper triangular part to 0
    
    # causal order
    P = np.eye(p)
    rng.shuffle(P)

    # mixing matrices
    A = P.T @ np.linalg.inv(np.eye(p) - B) @ P

    # observations
    X = np.array([Ai @ Si for Ai, Si in zip(A, S + N)])

    return X, P, B, A


def run_experiment(m, p, n, noise_level, nb_gaussian_sources, random_state, ica_algo):
    rng = np.random.RandomState(random_state)
    # generate observations X, causal order P, and causal effects B
    X, P, B, A = sample_data(m, p, n, noise_level, nb_gaussian_sources, rng)

    # apply either our method, Multi Group DirectLiNGAM, or LiNGAM
    if ica_algo in ["multiviewica", "shica_j", "shica_ml"]:
        # apply our main function to retrieve P, B, and W
        P_estimate, B_estimates, _, _, W_estimates = mvica_lingam(
            X, ica_algo=ica_algo, random_state=random_state)
    elif ica_algo == "multi_group_direct_lingam":
        # apply Multi Group DirectLiNGAM to retrieve P, B, and W
        model = lingam.MultiGroupDirectLiNGAM()
        model.fit(list(np.swapaxes(X, 1, 2)))
        # causal order P
        P_estimate = np.eye(p)[model.causal_order_]
        # causal effect matrices B
        B_s_estimates = np.array(model.adjacency_matrices_)
        B_estimates = P_estimate @ B_s_estimates @ P_estimate.T
        # reconstruct what would be unmixing matrices W
        W_estimates = P_estimate.T @ (np.eye(p) - B_estimates) @ P_estimate
    elif ica_algo == "lingam":
        # apply LiNGAM to retrieve P, B, and W
        P_estimates = []
        B_s_estimates = []
        B_estimates = []
        model = lingam.ICALiNGAM()
        for i in range(m):
            model.fit(np.swapaxes(X[i], 0, 1))    
            # causal order P
            P_estimate = np.eye(p)[model.causal_order_]
            P_estimates.append(P_estimate)
            # causal effect matrix B
            B_s_estimate = np.array(model._adjacency_matrix)
            B_s_estimates.append(B_s_estimate)
            B_estimate = P_estimate @ B_s_estimate @ P_estimate.T
            B_estimates.append(B_estimate)
        P_estimates = np.array(P_estimates)  # shape (m, p, p) and not (p, p)
        B_s_estimates = np.array(B_s_estimates)
        B_estimates = np.array(B_estimates)
        # reconstruct unmixing matrices W
        W_estimates = np.array(
            [Pi.T @ I_Bi @ Pi for Pi, I_Bi in zip(P_estimates, np.eye(p) - B_estimates)])
    else:
        raise ValueError("Wrong ica_algo.")
    
    # errors
    if ica_algo != "lingam":
        error_P = 1 - (P_estimate == P).all()
    else:
        error_P = np.mean([1 - (Pi == P).all() for Pi in P_estimates])
    error_B = np.mean((B_estimates - B) ** 2)
    amari = np.mean([amari_distance(Wi, Ai) for Wi, Ai in zip(W_estimates, A)])
    
    # output
    output = {
        "ica_algo": ica_algo,
        "nb_gaussian_sources": nb_gaussian_sources,
        "noise_level": noise_level,
        "random_state": random_state,
        "error_P": error_P,
        "error_B": error_B,
        "amari_distance": amari,
    }
    return output


# fixed parameters
m = 5
p = 4
n = 1000
N_JOBS = 5

# varying parameters
nb_gaussian_sources_list = [0, 2, 4]
nb_seeds = 50
random_state_list = np.arange(nb_seeds)
noise_level_list = np.logspace(-2, 2, 21)
algo_list = ["multiviewica", "shica_j", "shica_ml", "multi_group_direct_lingam", "lingam"]

# run experiment
nb_expes = len(nb_gaussian_sources_list) * len(random_state_list) * len(noise_level_list) \
    * len(algo_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        noise_level=noise_level,
        nb_gaussian_sources=nb_gaussian_sources,
        random_state=random_state,
        ica_algo=ica_algo,
    ) for noise_level, nb_gaussian_sources, random_state, ica_algo
    in product(noise_level_list, nb_gaussian_sources_list, random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/results/noise_in_xaxis/"
# results_dir = "/Users/ambroiseheurtebise/Desktop/mvica_lingam/experiments/results/fig2_shica/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
