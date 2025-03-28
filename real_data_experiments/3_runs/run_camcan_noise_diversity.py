# %%
import numpy as np
import pickle
from time import time
from pathlib import Path
import os
from shica import shica_ml
import matplotlib.pyplot as plt


# Limit the number of jobs
N_JOBS = 4
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1 "
    "--xla_force_host_platform_device_count=1"
)

# Parameters
n_subjects = 152
parcellation = "aparc_sub"
n_labels = 38
subset = 15  # or None
ica_algo = "shica_ml"
random_state = 42
whitening = False

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/MICaDo/real_data_experiments")
load_dir = expes_dir / f"2_data_envelopes/{parcellation}_{n_subjects}_subjects"

X_loaded = np.load(load_dir / f"X.npz")
X_list = [X_loaded[key] for key in X_loaded.files]

# Load labels
with open(load_dir / f"labels.pkl", "rb") as f:
    labels_list = pickle.load(f)

# Get all 38 labels
i = 0
while len(labels_list[i]) < n_labels:
    i+=1
labels = labels_list[i]
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Keep only 10 regions
selected_label_names = [
    'superiortemporal_3-lh',
    'superiortemporal_5-rh',
    'pericalcarine_1-lh',
    'pericalcarine_4-rh',
    'postcentral_6-lh',
    'postcentral_8-lh',
    'postcentral_7-rh',
    'postcentral_8-rh',
    'precentral_11-lh',
    'precentral_7-rh',
]

# Only keep subjects who have all these regions available
X = []
for X_current, labels_current in zip(X_list, labels_list):
    label_names_current = {label.name for label in labels_current}
    if all(name in label_names_current for name in selected_label_names):
        label_to_row = {label.name: row for label, row in zip(labels_current, X_current)}
        filtered_X = np.array([label_to_row[name] for name in selected_label_names])
        X.append(filtered_X)
X = np.array(X)  # shape (98, 10, 1760)
labels = [label for label in labels if label.name in selected_label_names]
n_subjects_full = len(X)

# whiten X
def whiten_data(X):
    n_views, n_components, n_timepoints = X.shape
    X_flat = X.reshape(n_views * n_components, n_timepoints)
    mean = X_flat.mean(axis=1, keepdims=True)
    X_centered = X_flat - mean
    cov = np.cov(X_centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-10)) @ eigvecs.T
    X_whitened = whitening_matrix @ X_centered
    return X_whitened.reshape(n_views, n_components, n_timepoints)

if whitening:
    X = whiten_data(X)

if subset:
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), replace=False, size=subset)
    X = X[idx]

# %%
# batch averaging of either X or S
def batch_avg_X(X, n_batches=40):
    m, p, n = X.shape
    X_res = np.zeros((m, p, n // n_batches))
    batch_size = n // n_batches
    for i in range(n_batches):
        X_res += X[:, :, i*batch_size: (i+1)*batch_size]
    X_res /= n_batches
    return X_res

def batch_avg_S(S, n_batches=40):
    p, n = S.shape
    S_res = np.zeros((p, n // n_batches))
    batch_size = n // n_batches
    for i in range(n_batches):
        S_res += S[:, i*batch_size: (i+1)*batch_size]
    S_res /= n_batches
    return S_res

X_avg = batch_avg_X(X)

# %%
# Apply ShICA-ML
start = time()
Sigmas_init = np.ones((X.shape[0], X.shape[1]))
W, Sigmas, S_avg = shica_ml(X, Sigmas_init=Sigmas_init, init=None)
execution_time = time() - start
print(f"The method took {execution_time:.2f} s.")

# Save data
save_dir = Path(expes_dir / f"4_results/noise_diversity_{ica_algo}")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "W.npy", W)
np.save(save_dir / "Sigmas.npy", Sigmas)
np.save(save_dir / "S_avg.npy", S_avg)

# %%
