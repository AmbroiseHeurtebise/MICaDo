import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
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

    return X, P, B


def run_experiment(m, p, n, nb_gaussian_sources, random_state, ica_algo):
    rng = np.random.RandomState(random_state)
    # generate observations X, causal order P, and causal effects B
    X, P, B = sample_data(m, p, n, nb_gaussian_sources, rng)

    # apply our main function to retrieve P and B
    P_estimate, B_estimates, _, _ = mvica_lingam(
        X, ica_algo=ica_algo, random_state=random_state)
    
    # errors
    error_P = 1 - (P_estimate == P).all()
    error_B = np.mean((B_estimates - B) ** 2)
    
    # output
    output = {
        "ica_algo": ica_algo,
        "nb_gaussian_sources": nb_gaussian_sources,
        "n": n,
        "random_state": random_state,
        "error_P": error_P,
        "error_B": error_B,
    }
    return output


# fixed parameters
m = 5
p = 4
N_JOBS = 2

# varying parameters
nb_gaussian_sources_list = [2]
nb_seeds = 2
random_state_list = np.arange(nb_seeds)
n_list = np.logspace(2, 4, 3, dtype=int)
algo_list = ["multiviewica", "shica_j", "shica_ml"]

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
# results_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/results/fig2_shica/"
results_dir = "/Users/ambroiseheurtebise/Desktop/mvica_lingam/experiments/results/fig2_shica/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
