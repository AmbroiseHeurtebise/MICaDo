# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pickle
from pathlib import Path


# %%
# Parameters
n_subjects = 103
parcellation = "aparc"
n_arrows = 10
only_clean = True

# Load results
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments")
if only_clean:
    results_dir = Path(expes_dir / f"4_results/{parcellation}_{n_subjects}_subjects_clean")
else:
    results_dir = Path(expes_dir / f"4_results/{parcellation}_{n_subjects}_subjects")
P = np.load(results_dir / "P.npy")
T = np.load(results_dir / "T.npy")
B = np.load(results_dir / "B.npy")
with open(results_dir / f"labels.pkl", "rb") as f:
    labels = pickle.load(f)

# %%
# Plot average matrix T (should be lower triangular)
plt.imshow(np.mean(np.abs(T), axis=0))
plt.colorbar()
plt.title("Average absolute value lower triangular matrix T")
plt.show()

# %%
# Normalize matrices Bi: divide each Bi by its max in absolute value
B_maxs = np.array([np.max(Bi_abs) for Bi_abs in np.abs(B)])[:, np.newaxis, np.newaxis]
B_norm = B / B_maxs
B_avg = np.mean(B_norm, axis=0)

# %%
# Choose random subject (bads: 12, 30, 62, 68, 88, 138)
idx = np.random.randint(0, n_subjects)
B_avg = B[idx]

# %%
# Plot average normalized adjacency matrix
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=np.min(B_avg), vmax=np.max(B_avg), vcenter=0)
plt.imshow(B_avg, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title("Average normalized adjacency matrix B")
label_names = [label.name for label in labels]
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
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

homogenise = False
if homogenise:
    B_avg_subset = np.sign(B_avg_subset) * np.sqrt(np.abs(B_avg_subset))
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=np.min(B_avg_subset), vmax=np.max(B_avg_subset), vcenter=0)
plt.imshow(B_avg_subset, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title(f"Average adjacency matrix B ({n_arrows} highest effects)")
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
ax.set_yticklabels(label_names)
plt.show()

# %%
order = np.argmax(P, axis=1)
labels_ordered = [label_names[order[i]] for i in range(len(label_names))]
labels_ordered

# %%
