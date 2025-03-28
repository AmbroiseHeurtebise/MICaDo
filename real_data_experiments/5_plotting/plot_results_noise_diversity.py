# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pickle
from pathlib import Path
from scipy.stats import kurtosis


# %%
# Parameters
ica_algo = "shica_ml"
threshold = 0.5
n_sources_kept = 3

# Load results
expes_dir = Path("/storage/store2/work/aheurteb/MICaDo/real_data_experiments")
results_dir = Path(expes_dir / f"4_results/noise_diversity_{ica_algo}")

W = np.load(results_dir / "W.npy")
Sigmas = np.load(results_dir / "Sigmas.npy")
S_avg = np.load(results_dir / "S_avg.npy")

p = len(S_avg)

# %%
# compute excess kurtosis
def gaussianity_score(S):
    scores = kurtosis(S, axis=1, fisher=True)
    return scores

scores = gaussianity_score(S_avg)

# %%
# keep nearly-Gaussian sources
# idx = np.where(np.abs(scores) < threshold)[0]
idx = np.argsort(np.abs(scores))[:n_sources_kept]

# pairs of nearly-Gaussian sources
pairs = [(idx[i], idx[j]) for i, j in zip(*np.triu_indices(len(idx), k=1))]
summed_scores = []
for pair in pairs:
    summed_scores.append(scores[pair[0]] + scores[pair[1]])
ordered_pairs = [pairs[i] for i in np.argsort(summed_scores)]

# %%
# compute differences of variances
diff = []
for pair in ordered_pairs:
    # the variance of source 3 of subject 0 is extremely high (1e7) so I removed this subject
    # that's why I start at 1
    sigma1 = Sigmas[1:, pair[0]]
    sigma2 = Sigmas[1:, pair[1]]
    diff.append(sigma1 - sigma2)
diff = np.array(diff)

# %%
