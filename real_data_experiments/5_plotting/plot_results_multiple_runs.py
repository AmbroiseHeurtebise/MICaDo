# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pickle
from pathlib import Path
from scipy.stats import spearmanr

# %%
# Parameters
n_subjects = 98
n_runs = 50
ica_algo = "shica_ml"
n_arrows = 10
random_state = 42
rng = np.random.RandomState(random_state)

# Load results
expes_dir = Path("/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments")
results_dir = Path(expes_dir / f"4_results/aparc_sub_{n_subjects}_subjects_{n_runs}_runs_{ica_algo}")

P_total = np.load(results_dir / "P_total.npy")
T_total = np.load(results_dir / "T_total.npy")
B_total = np.load(results_dir / "B_total.npy")
with open(results_dir / f"labels.pkl", "rb") as f:
    labels = pickle.load(f)

# %%
# matplotlib style
fontsize = 20
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# %%
# Compute Spearman rank coefficients
spearmanr_matrix = np.zeros((n_runs, n_runs))
for i in range(n_runs):
    for j in range(n_runs):
        B1_median = np.median(B_total[i], axis=0)
        B2_median = np.median(B_total[j], axis=0)
        rho, p_value = spearmanr(B1_median.flatten(), B2_median.flatten())
        spearmanr_matrix[i, j] = rho
np.fill_diagonal(spearmanr_matrix, 0)

# Compute the average Spearman rank correlation
upper_triangular_values = spearmanr_matrix[np.triu_indices(n_runs, k=1)]
avg_corr = np.mean(upper_triangular_values)

# Plot obtained Spearman rank correlations
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
plt.imshow(spearmanr_matrix, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title(f"Average = {avg_corr:.2f}")
# ax.set_xticks(np.arange(n_runs))
# ax.set_yticks(np.arange(n_runs))
ax.set_xlabel("Runs")
ax.set_ylabel("Runs")

save = True
if save:
    figures_dir = "/storage/store2/work/aheurteb/mvica_lingam/real_data_experiments/6_figures//"
    plt.savefig(figures_dir + f"spearmanr_coefs.pdf", bbox_inches="tight")
plt.show()

# %%
# Plot median matrix T (should be lower triangular)
T_avg = np.mean(np.abs(T_total), axis=(0, 1))
plt.imshow(T_avg)
plt.colorbar()
plt.title("Average absolute value lower triangular matrix T")
plt.show()

# %%
# Plot median adjacency matrix
B_median = np.median(B_total, axis=(0, 1))
fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=np.min(B_median), vmax=np.max(B_median), vcenter=0)
plt.imshow(B_median, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title("Median adjacency matrix B")
label_names = [label.name for label in labels]
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
ax.set_yticklabels(label_names)
plt.show()

# %%
# Only keep the most important effects
M = np.abs(B_median)
indices = np.argsort(M.flatten())[::-1]
ranked_flat = np.zeros(M.size)
ranked_flat[indices] = np.arange(M.size)
B_avg_rank = ranked_flat.reshape(M.shape)
B_avg_subset = B_median * (B_avg_rank < n_arrows)

fig, ax = plt.subplots()
norm = TwoSlopeNorm(vmin=np.min(B_avg_subset), vmax=np.max(B_avg_subset), vcenter=0)
plt.imshow(B_avg_subset, norm=norm, cmap="coolwarm")
plt.colorbar()
plt.title(f"Median adjacency matrix B ({n_arrows} highest effects)")
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names, rotation=45, ha="right")
ax.set_yticklabels(label_names)
plt.show()

# %%
# random select 6 subjects
random_state = 46
rng = np.random.RandomState(random_state)
idx = rng.choice(len(B_total), size=6, replace=False)
idx[2] = 44
B_median_2 = np.median(B_total, axis=(0))  # shape (49, 10, 10)
B_subset = B_median_2[idx]

# %%
# Plot median adjacency matrix
def plot_B(B_median, n_arrows=10):
    # Only keep the most important effects
    M = np.abs(B_median)
    indices = np.argsort(M.flatten())[::-1]
    ranked_flat = np.zeros(M.size)
    ranked_flat[indices] = np.arange(M.size)
    B_avg_rank = ranked_flat.reshape(M.shape)
    B_avg_subset = B_median * (B_avg_rank < n_arrows)
    
    fig, ax = plt.subplots()
    if np.min(B_avg_subset) < 0:
        norm = TwoSlopeNorm(vmin=np.min(B_avg_subset), vmax=np.max(B_avg_subset), vcenter=0)
    else:
        norm = TwoSlopeNorm(
            vmin=np.min(B_avg_subset), vmax=np.max(B_avg_subset),
            vcenter=(np.min(B_avg_subset)+np.max(B_avg_subset))/2)
    plt.imshow(B_avg_subset, norm=norm, cmap="coolwarm")
    plt.colorbar()
    plt.title("Median adjacency matrix B")
    label_names = [label.name for label in labels]
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    plt.show()

# %%
# plot 6 causal effect matrices
for i in range(6):
    plot_B(B_subset[i])

# %%
# for i in range(n_subjects // 2):
#     print(i)
#     plot_B(B_median_2[i])

# %%
