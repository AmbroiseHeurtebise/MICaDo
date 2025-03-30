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
nb_seeds = 20
metrics = ["error_B_abs", "error_T_abs", "error_P_spearmanr"]  # or "error_B", "error_P_exact", "amari_distance"

# read dataframe
results_dir = "/storage/store2/work/aheurteb/MICaDo/simulation_studies/results/results_sparsity_of_Ti/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# metric names
metric_names = []
for metric in metrics:
    if metric == "error_B" or metric == "error_B_abs":
        metric_names.append(r"Error on $B^i$")
    elif metric == "error_T" or metric == "error_T_abs":
        metric_names.append(r"Error on $T^i$")
    elif metric == "error_P_exact":
        metric_names.append(r"Error on $P$")
    elif metric == "error_P_spearmanr":
        metric_names.append("Spearman's rank\ncorrelation on" + r" $P$")
    elif metric == "amari_distance":
        metric_names.append("Amari distance")

# labels, dashes, curves order and titles
labels = ['MICaDo-ML', 'MICaDo-J']
dashes = ['', '']
hue_order = ["shica_ml", "shica_j"]
titles = ["Multiple causal orderings", "Shared causal ordering"]

# subplots
fig, axes = plt.subplots(3, 2, figsize=(8.5, 6), sharex=True)
for i, ax in enumerate(axes.flat):
    metric = metrics[i // 2]
    shared_causal_ordering = bool(i % 2)
    data = df[df["shared_causal_ordering"] == shared_causal_ordering]
    sns.lineplot(
        data=data, x="nb_zeros_Ti", y=metric, linewidth=2.5, hue="ica_algo", estimator=np.median,
        errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo", ax=ax,
        dashes=dashes, markers=True)
    if i // 2 != 2:
        ax.set_yscale("log")
    # if i % 2 == 0:
    ax.set_ylabel(metric_names[i // 2])
    # title, grid, and legend
    if i // 2 == 0:
        ax.set_title(titles[i], fontsize=fontsize)
    ax.set_xlabel("")
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
xlabel = fig.supxlabel(r"# sparse entries in each $T^i$", fontsize=fontsize)
xlabel.set_position((0.5, 0.07))
plt.gcf().align_labels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)

# legend
palette = sns.color_palette()[:2]
legend_styles = [
    Line2D([0], [0], color=palette[0], linewidth=2.5, linestyle='-', marker='o', 
           markeredgecolor="white", markersize=6),
    Line2D([0], [0], color=palette[1], linewidth=2.5, linestyle='-', marker='X', 
           markeredgecolor="white", markersize=7),
]
fig.legend(
    legend_styles, labels, bbox_to_anchor=(0.5, 1.02), loc="center",
    ncol=2, borderaxespad=0., fontsize=fontsize
)

# save figure
figures_dir = Path("/storage/store2/work/aheurteb/MICaDo/simulation_studies/figures")
plt.savefig(figures_dir / f"simulation_sparsity_of_Ti.pdf", bbox_inches="tight")
plt.show()
