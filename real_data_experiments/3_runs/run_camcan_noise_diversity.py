# %%
import numpy as np
import pickle
from time import time
from pathlib import Path
import os
from shica import shica_ml


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
subset = False
ica_algo = "shica_ml"
random_state = 42

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/MICaDo/real_data_experiments")
load_dir = expes_dir / f"2_data_envelopes/{parcellation}_{n_subjects}_subjects"

X_loaded = np.load(load_dir / f"X.npz")
X_list = [X_loaded[key] for key in X_loaded.files]

# %%
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

# %%
# Apply ShICA-ML
start = time()
W, Sigmas, S_avg = shica_ml(X)
execution_time = time() - start
print(f"The method took {execution_time:.2f} s.")

# %%
# Save data
save_dir = Path(expes_dir / f"4_results/noise_diversity_{ica_algo}")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "W.npy", W)
np.save(save_dir / "Sigmas.npy", Sigmas)
np.save(save_dir / "S_avg.npy", S_avg)
