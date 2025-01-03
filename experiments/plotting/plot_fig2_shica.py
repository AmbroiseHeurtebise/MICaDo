import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
nb_seeds = 50
nb_gaussian_sources_list = [4, 0, 2]
errors = ["amari_distance", "error_B", "error_P"]
error_names = ["Amari distance", "Error on B", "Error rate on P"]
titles = ["Gaussian", "Non-Gaussian", "Half-G / Half-NG"]
estimator = "mean"

# read dataframe
results_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/results/fig2_shica/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharex="col", sharey="row")
for i, ax in enumerate(axes.flat):
    # number of Gaussian sources; one for each of the 3 columns
    nb_gaussian_sources = nb_gaussian_sources_list[i % 3]
    data = df[df["nb_gaussian_sources"] == nb_gaussian_sources]
    # error; one for each of the 3 rows
    y = errors[i // 3]
    # subplot
    if i // 3 != 2 and estimator == "median":
        sns.lineplot(
            data=data, x="n", y=y, linewidth=2.5, hue="ica_algo", estimator=np.mean,
            ax=ax, errorbar=lambda x: (np.quantile(x, 0.025), np.quantile(x, 0.975)))
    else:
        sns.lineplot(
            data=data, x="n", y=y, linewidth=2.5, hue="ica_algo", ax=ax, errorbar=('ci', 95))
    # set axis in logscale, except for the yaxis of the middle row
    ax.set_xscale("log")
    if i // 3 != 2:
        ax.set_yscale("log")
    # ylabel
    ax.set_xlabel("")
    ax.set_ylabel("")
    if i % 3 == 0:
        ax.set_ylabel(error_names[i // 3])
    # title, grid, and legend
    if i // 3 == 0:
        ax.set_title(titles[i], fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
label = fig.supxlabel("Number of samples", fontsize=fontsize)
label.set_position((0.5, 0.055))
plt.gcf().align_labels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
# legend
handles, labels = ax.get_legend_handles_labels()
labels = ['MVICA-LiNGAM', 'ShICA-J-LiNGAM', 'ShICA-ML-LiNGAM', 'MultiGroupDirectLiNGAM']
fig.legend(
    handles, labels, bbox_to_anchor=(0.5, 1.02), loc="center",
    ncol=len(labels), borderaxespad=0., fontsize=fontsize)

# save figure
figures_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/figures/"
plt.savefig(figures_dir + "fig2_shica.pdf", bbox_inches="tight")
plt.show()
