import pandas as pd
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
nb_seeds = 10
nb_gaussian_sources_list = [4, 0, 2]
error = "error_B"  # ["amari_distance", "error_B", "error_P"]
error_name = "Error on B"  # ["Amari distance", "Error on B", "Error rate on P"]
titles = ["Gaussian", "Non-Gaussian", "Half-G / Half-NG"]
labels = [
    'MVICA-LiNGAM', 'ShICA-J-LiNGAM', 'ShICA-ML-LiNGAM', 'MultiGroupDirectLiNGAM',
    'LiNGAM']

# read dataframe
results_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/results/noise_in_xaxis/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey="row")
for i, ax in enumerate(axes.flat):
    # number of Gaussian sources; one for each of the 3 columns
    nb_gaussian_sources = nb_gaussian_sources_list[i % 3]
    data = df[df["nb_gaussian_sources"] == nb_gaussian_sources]
    # subplot
    sns.lineplot(
        data=data, x="noise_level", y=error, linewidth=2.5, hue="ica_algo", ax=ax, errorbar=('ci', 95))
    ax.set_xscale("log")
    if error != "error_P":
        ax.set_yscale("log")
    if i == 0 and error == "error_B":
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 1e5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if i == 0:
        ax.set_ylabel(error_name)
    # title, grid, and legend
    ax.set_title(titles[i], fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
label = fig.supxlabel("Noise level", fontsize=fontsize)
label.set_position((0.5, 0.11))
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
# legend
handles, _ = ax.get_legend_handles_labels()
fig.legend(
    handles, labels, bbox_to_anchor=(0.5, 1.09), loc="center",
    ncol=3, borderaxespad=0., fontsize=fontsize)

# save figure
figures_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/figures/"
plt.savefig(figures_dir + "simulation_noise_in_xaxis.pdf", bbox_inches="tight")
plt.show()
