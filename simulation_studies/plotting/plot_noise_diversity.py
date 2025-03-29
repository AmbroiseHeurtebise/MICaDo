# %%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


# matplotlib style
fontsize = 15
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# parameters 
nb_seeds = 2
metric = "error_B"  # or "error_T", "error_P_exact", "error_P_spearmanr", "amari_distance"
# include_multiviewica = True

# read dataframe
results_dir = "/storage/store2/work/aheurteb/MICaDo/simulation_studies/results/results_noise_diversity/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# %%
sns.lineplot(
    data=df, x="nb_equal_variances", y=metric, linewidth=2.5, hue="ica_algo", estimator=np.median,
    errorbar=('ci', 95), style="ica_algo",
    markers=True)

# %%
