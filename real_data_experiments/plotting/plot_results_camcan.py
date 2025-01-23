# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pickle
from pathlib import Path


# %%
# Parameters
n_subjects_total = 40
n_subjects_used = 40
parcellation = "aparc"
hemi = "lh"
n_arrows = 7

# Load labels
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments")
labels_dir = expes_dir / "data_envelopes"
with open(labels_dir / f"labels_{parcellation}_{n_subjects_total}_subjects.pkl", "rb") as f:
    labels_total = pickle.load(f)
if hemi == "both":
    labels = labels_total
else:
    labels = [label for label in labels_total if label.hemi == hemi]

# Load P, T, and B
results_dir = Path(expes_dir / f"results/{parcellation}_{n_subjects_used}_subjects_{hemi}")
P = np.load(results_dir / "P.npy")
T = np.load(results_dir / "T.npy")
B = np.load(results_dir / "B.npy")

# %%
# Plot average matrix T (should be lower triangular)
plt.imshow(np.mean(np.abs(T), axis=0))
plt.colorbar()
plt.title("Average absolute value lower triangular matrix T")
plt.show()

# %%
# Plot average adjacency matrix
fig, ax = plt.subplots()
B_avg = np.mean(B, axis=0)
norm = TwoSlopeNorm(vmin=np.min(B_avg), vmax=np.max(B_avg), vcenter=0)
plt.imshow(B_avg, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title("Average adjacency matrix B")
label_names = [label.name for label in labels]
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
ax.set_xticklabels(label_names, rotation=45)
ax.set_yticklabels(label_names)
plt.show()

# %%
# Only keep the most important effects
M = np.abs(B_avg)
indices = np.argsort(M.flatten())[::-1]
ranked_flat = np.zeros(M.size)
ranked_flat[indices] = np.arange(M.size)
B_avg_rank = ranked_flat.reshape(M.shape)
B_avg_subset = B_avg * (B_avg_rank < n_arrows)

fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=np.min(B_avg_subset), vmax=np.max(B_avg_subset), vcenter=0)
plt.imshow(B_avg_subset, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title(f"Average adjacency matrix B ({n_arrows} highest effects)")
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
ax.set_xticklabels(label_names, rotation=45)
ax.set_yticklabels(label_names)
plt.show()

# %%
