import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


# matplotlib style
fontsize = 14
rc = {
    "font.size": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "serif",
}
plt.rcParams.update(rc)

# parameters 
nb_seeds = 50
metric = "error_P_spearmanr"  # or "error_B", "error_P_exact", "amari_distance"

# read dataframe
results_dir = "/storage/store2/work/aheurteb/MICaDo/simulation_studies/results/results_sparsity_of_Ti/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# metric names
if metric == "error_B" or metric == "error_B_abs":
    metric_name = r"Error on $B^i$"
elif metric == "error_T" or metric == "error_T_abs":
    metric_name = r"Error on $T^i$"
elif metric == "error_P_exact":
    metric_name = r"Error on $P$"
elif metric == "error_P_spearmanr":
    metric_name = "Spearman's rank\ncorrelation of the\ncausal ordering(s)"
elif metric == "amari_distance":
    metric_name = "Amari distance"

# labels, dashes, curves order and titles
labels = ['MICaDo-ML']
dashes = ['']
hue_order = ["shica_ml"]
titles = ["Multiple causal orderings", "Shared causal ordering"]

# subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    shared_causal_ordering = bool(i % 2)
    data = df[df["shared_causal_ordering"] == shared_causal_ordering]
    sns.lineplot(
        data=data, x="nb_zeros_Ti", y=metric, linewidth=2.5, hue="ica_algo", estimator=np.median,
        errorbar=('ci', 95), hue_order=hue_order, style_order=hue_order, style="ica_algo", ax=ax,
        dashes=dashes, markers=True)
    ax.set_xlabel("")
    if i == 0:
        ylabel = ax.set_ylabel(metric_name)
        ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_title(titles[i], fontsize=fontsize)
    ax.grid(which='both', linewidth=0.5, alpha=0.5)
    ax.get_legend().remove()
# ax.set_ylim(-1.1, 1.1)
# ax.set_yticks([-1, 0, 1])
# ax.set_yticklabels([-1, 0, 1])
ax.set_yticks([0, 1])
ax.set_yticklabels([0, 1])
xlabel = fig.supxlabel(r"# sparse entries in each $T^i$", fontsize=fontsize)
xlabel.set_position((0.5, 0.16))
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
    legend_styles, labels, bbox_to_anchor=(0.5, 1.03), loc="center",
    ncol=2, borderaxespad=0., fontsize=fontsize
)

# caption
caption = (
    "Caption: Data are generated with $m=8$ views and $p=6$ disturbances, consisting \n"
    "of 2 sub-Gaussian, 2 Gaussian, and 2 super-Gaussian disturbances. The x-axis \n"
    "represents the number of sparse entries in the strictly lower triangular part of \n"
    "each " + r"$T^i$" + ", while the y-axis shows the Spearman's rank correlation between true and \n"
    "estimated causal orderings. Results justify Assumptions 2 and 3, since recovering \n"
    "the causal ordering is significantly easier in the shared causal ordering scenario \n"
    "(Assumption 3) than in the multiple causal orderings scenario (Assumption 2)."
)
fig.text(0.5, -0.2, caption, ha='center', va='center', fontsize=fontsize)

# save figure
figures_dir = Path("/storage/store2/work/aheurteb/MICaDo/simulation_studies/figures")
plt.savefig(figures_dir / f"simulation_sparsity_of_Ti.pdf", bbox_inches="tight")
plt.show()
