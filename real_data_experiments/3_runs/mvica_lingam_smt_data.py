import numpy as np
import pickle
from time import time
from pathlib import Path
import os
from mvica_lingam.mvica_lingam import mvica_lingam


# Limit the number of jobs
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Parameters
n_subjects = 154
parcellation = "aparc"
n_labels = 10
subset = True

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments")
load_dir = expes_dir / f"2_data_envelopes/{parcellation}_{n_subjects}_subjects"

X_loaded = np.load(load_dir / f"X_{parcellation}_{n_subjects}_subjects.npz")
X_list = [X_loaded[key] for key in X_loaded.files]

# Load labels
with open(load_dir / f"labels_{parcellation}_{n_subjects}_subjects.pkl", "rb") as f:
    labels_list = pickle.load(f)

# Only keep subjects that have ``n_labels`` labels
X = []
for i in range(len(labels_list)):
    if len(labels_list[i]) == n_labels:
        X.append(X_list[i])
        labels = labels_list[i]
X = np.array(X)
n_subjects_full = len(X)  # may be lower than n_subjects

# Remove subjects who have a strange envelope
if subset and parcellation == "aparc":
    idx_remove = [
        8, 11, 17, 32, 36, 66, 71, 95, 99, 126, 133, 146, 3, 5, 6, 7, 9, 10, 23, 27, 28, 31, 33, 37, 38,
        44,47, 57, 65, 76, 78, 79, 96, 97, 105, 106, 112, 116, 117, 119, 125, 128, 132, 141, 143, 145]
    idx = list(set(np.arange(n_subjects_full)) - set(idx_remove))
    X = X[idx]
    n_subjects_full = len(X)

# Apply our method
start = time()
P, T, _, _, _ = mvica_lingam(X)
execution_time = time() - start
print(f"The method took {execution_time:.2f} s.")

# Recover the adjacency matrix
B = P.T @ T @ P

# Save data
save_dir = Path(expes_dir / f"4_results/{parcellation}_{n_subjects_full}_subjects")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "P.npy", P)
np.save(save_dir / "T.npy", T)
np.save(save_dir / "B.npy", B)
with open(save_dir / f"labels.pkl", "wb") as f:
    pickle.dump(labels, f)
