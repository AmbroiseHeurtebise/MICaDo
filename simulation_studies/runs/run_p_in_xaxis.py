import numpy as np
import pandas as pd
import os
from itertools import product
from joblib import Parallel, delayed
from utils import run_experiment


# limit number of jobs
N_JOBS = 4
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

# fixed parameters
n = 1000
noise_level = 1.
density = "sub_gauss_super"

# varying parameters
m_list = [3, 5, 8, 12, 16, 20]
p_list = [3, 6, 9]
nb_seeds = 2
random_state_list = np.arange(nb_seeds)
algo_list = ["multiviewica", "shica_j", "shica_ml", "multi_group_direct_lingam", "lingam"]

# run experiment
nb_expes = len(m_list) * len(p_list) * len(random_state_list) * len(algo_list)
print(f"\nTotal number of experiments : {nb_expes}")
print("\n###################################### Start ######################################")
dict_res = Parallel(n_jobs=N_JOBS)(
    delayed(run_experiment)(
        m=m,
        p=p,
        n=n,
        density=density,
        random_state=random_state,
        ica_algo=ica_algo,
        noise_level=noise_level,
    ) for m, p, random_state, ica_algo
    in product(m_list, p_list, random_state_list, algo_list)
)
print("\n################################ Obtained DataFrame ################################")
df = pd.DataFrame(dict_res)
print(df)

# save dataframe
results_dir = "/storage/store2/work/aheurteb/MICaDo/simulation_studies/results/results_p_in_xaxis/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df.to_csv(save_path, index=False)
print("\n####################################### End #######################################")
