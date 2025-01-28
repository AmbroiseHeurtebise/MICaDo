import pandas as pd
import numpy as np
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
nb_seeds = 3
errors = ["error_B", "error_T", "error_P"]  # ["amari_distance"]
error_names = ["Error on B", "Error on T", "Error rate on P"]  # ["Amari distance"]
estimator = "mean"
labels = ['ShICA-ML-LiNGAM', 'ShICA-J-LiNGAM', 'LiNGAM', 'MultiGroupDirectLiNGAM']
hue_order = ["shica_ml", "shica_j", "lingam", "multi_group_direct_lingam"]

# read dataframe
results_dir = "/storage/store2/work/aheurteb/mvica_lingam/simulation_studies/results/"
parent_dir = "gaussian_sources_in_xaxis/"
save_name = f"/DataFrame_with_{nb_seeds}_seeds_and_4_metrics"
save_path = results_dir + parent_dir + save_name
df = pd.read_csv(save_path)

# subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 2), sharex=True)
for i, ax in enumerate(axes.flat):
    y = errors[i // 3]
    sns.lineplot(
        data=df, x="nb_gaussian_sources", y=y, linewidth=2.5, hue="ica_algo", ax=ax, 
        errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo",
        dashes=['', '', (2, 2), (2, 2)])
    if i != 2:
        ax.set_yscale("log")
    ax.set_title(error_names[i], fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
label = fig.supxlabel("Number of samples", fontsize=fontsize)
label.set_position((0.5, 0.055))
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
# legend
palette = sns.color_palette()[:4]
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle='-'),
    Line2D([0], [0], color=palette[2], linewidth=2.5, linestyle='--'),
    Line2D([0], [0], color=palette[3], linewidth=2.5, linestyle='--'),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.03), loc="center",
    ncol=4, borderaxespad=0., fontsize=fontsize
)

# save figure
figures_dir = "/storage/store2/work/aheurteb/mvica_lingam/simulation_studies/figures/"
plt.savefig(figures_dir + f"simulation_gaussian_sources.pdf", bbox_inches="tight")
plt.show()
