import numpy as np
import pickle
from time import time
from pathlib import Path
from mvica_lingam.mvica_lingam import mvica_lingam


# Parameters
n_subjects_total = 40
n_subjects_used = 40
parcellation = "aparc"
hemi = "rh"

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments")
load_dir = expes_dir / "data_envelopes"
X = np.load(load_dir / f"X_{parcellation}_{n_subjects_total}_subjects.npy")
with open(load_dir / f"labels_{parcellation}_{n_subjects_total}_subjects.pkl", "rb") as f:
    labels = pickle.load(f)

# Separate the two hemispheres
idx_lh = [i for i in range(len(labels)) if labels[i].hemi == 'lh']
idx_rh = np.setdiff1d(np.arange(len(labels)), idx_lh)
X_lh = X[:, idx_lh]
X_rh = X[:, idx_rh]
labels_lh = [labels[i] for i in idx_lh]
labels_rh = [labels[i] for i in idx_rh]

# Choose either both hemispheres or only one of them
if hemi == "lh":
    X_used = X_lh
elif hemi == "rh":
    X_used = X_rh
else:
    X_used = X

# Apply our method
start = time()
P, T, _, _, _ = mvica_lingam(X_used[:n_subjects_used])
execution_time = time() - start
print(f"The method took {execution_time:.2f} s.")

# Recover the adjacency matrix
B = P.T @ T @ P

# Save data
save_dir = Path(expes_dir / f"results/{parcellation}_{n_subjects_used}_subjects_{hemi}")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "P.npy", P)
np.save(save_dir / "T.npy", T)
np.save(save_dir / "B.npy", B)
