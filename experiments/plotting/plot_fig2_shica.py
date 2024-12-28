import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# parameters 
nb_seeds = 50
nb_gaussian_sources_list = [0, 2, 4]
errors = ["amari_distance", "error_P", "error_B"]
algo_list = ["multiviewica", "shica_j", "shica_ml"]

# read dataframe
results_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/results/fig2_shica/"
save_name = f"DataFrame_with_{nb_seeds}_seeds"
save_path = results_dir + save_name
df = pd.read_csv(save_path)

# colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    # number of Gaussian sources; one for each of the 3 columns
    nb_gaussian_sources = nb_gaussian_sources_list[i // 3]
    data = df[df["nb_gaussian_sources"] == nb_gaussian_sources]
    # error; one for each of the 3 rows
    y = errors[i % 3]
    # subplot
    for j, algo in enumerate(algo_list):
        sns.lineplot(
            data=data, x="n", y=y, linewidth=2.5,
            label=algo, estimator=np.median, c=colors[j])
    ax.set_xscale("log")
    ax.set_yscale("log")
    if i // 3 == 0:
        ax.set_ylabel(errors[i % 3])
    ax.grid()
    ax.legend_.remove()
fig.supxlabel("Number of samples")
# legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels, bbox_to_anchor=(0.5, 1.05), loc="center",
    ncol=3, borderaxespad=0.)

# save figure
figures_dir = "/storage/store2/work/aheurteb/mvica_lingam/experiments/figures/"
plt.savefig(figures_dir + "fig2_shica.pdf")
plt.show()
