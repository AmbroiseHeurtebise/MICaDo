import numpy as np
import pickle
from pathlib import Path
import os
from mvica_lingam.mvica_lingam import mvica_lingam


# Limit the number of jobs
N_JOBS = 4
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

# Parameters
n_runs = 50
keep_subjects_rate = 1 / 2  # only keep 50% of the subjects
ica_algo = "shica_ml"

# Load data
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments")
load_dir = expes_dir / f"2_data_envelopes/aparc_sub_152_subjects"

X_loaded = np.load(load_dir / f"X.npz")
X_list = [X_loaded[key] for key in X_loaded.files]

# Load labels
with open(load_dir / f"labels.pkl", "rb") as f:
    labels_list = pickle.load(f)

# Get all 38 labels
i = 0
while len(labels_list[i]) < 38:
    i+=1
labels = labels_list[i]

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
n_labels = len(selected_label_names)

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

# Run our method ``n_runs`` times
n_subjects_batch = int(n_subjects_full * keep_subjects_rate)
B_total = np.zeros((n_runs, n_subjects_batch, n_labels, n_labels))
T_total = np.zeros((n_runs, n_subjects_batch, n_labels, n_labels))
P_total = np.zeros((n_runs, n_labels, n_labels))
for i in range(n_runs):
    print(f"Run number {i} / {n_runs}")
    rng = np.random.RandomState(i)
    # Select of subset of subjects
    subjects_idx = rng.choice(n_subjects_full, size=n_subjects_batch, replace=False)
    X_subset = X[subjects_idx]
    # Apply our method
    B, T, P, _, _ = mvica_lingam(
        X_subset, ica_algo=ica_algo, random_state=i,
        new_find_order_function=False)
    B_total[i] = B
    T_total[i] = T
    P_total[i] = P

# Save data
save_dir = Path(expes_dir / f"4_results/aparc_sub_{n_subjects_full}_subjects_{n_runs}_runs_{ica_algo}")
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / "B_total.npy", B_total)
np.save(save_dir / "T_total.npy", T_total)
np.save(save_dir / "P_total.npy", P_total)
with open(save_dir / f"labels.pkl", "wb") as f:
    pickle.dump(labels, f)
