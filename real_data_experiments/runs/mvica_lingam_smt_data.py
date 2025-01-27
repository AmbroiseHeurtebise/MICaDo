import numpy as np
import pickle
from time import time
from pathlib import Path
from mvica_lingam.mvica_lingam import mvica_lingam


# Parameters
n_subjects = 3
parcellation = "aparc"
n_labels = 10

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments")
load_dir = expes_dir / "data_envelopes"
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

# Apply our method
start = time()
P, T, _, _, _ = mvica_lingam(X)
execution_time = time() - start
print(f"The method took {execution_time:.2f} s.")

# Recover the adjacency matrix
B = P.T @ T @ P

# Save data
save_dir = Path(expes_dir / f"results/{parcellation}_{n_subjects}_subjects")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "P.npy", P)
np.save(save_dir / "T.npy", T)
np.save(save_dir / "B.npy", B)
with open(save_dir / f"labels_{parcellation}_{n_subjects}_subjects.pkl", "wb") as f:
    pickle.dump(labels, f)
