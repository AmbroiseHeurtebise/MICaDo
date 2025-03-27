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
n_sources_kept = 5

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
idx = np.where(np.abs(scores) < threshold)[0]

idx = np.argmin(scores)[:n_sources_kept]