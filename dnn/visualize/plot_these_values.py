import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    gen_seeds = [80346, 80346, 80346, 793, 16, 16, 2021, 4, 4]
    gen_sizes = [1000, 1150, 1290, 1420, 1600, 1790, 1910, 1970, 2380]
    seeds_tried = [24, 105, 418, 666, 3309, 42, 190, 762, 7, 3000]

    # MTP to MTP
    # our_pop_accs = np.array([
    #     [0.720, 0.728, 0.582, 0.765, 0],
    #     [0.640, 0.663, 0.478, 0.698, 0],
    #     [0.467, 0.499, 0, 0.513, 0],
    #     [0.409, 0.453, 0, 0.459, 0],
    #     [0.370, 0.422, 0, 0.425, 0]
    # ])
    # flip_pop_accs = np.array([
    #     [[0.952, 0.952, 0.952], [0.947, 0.947, 0.947], [0.894, 0.892, 0.894],
    #         [0.958, 0.958, 0.956], [0.966, 0.966, 0.965]],
    #     [[0.944, 0.945, 0.945], [0.938, 0.938, 0.940], [0.875, 0.877, 0.877],
    #         [0.947, 0.949, 0.949], [0.944, 0.944, 0.949]],
    #     [[0.907, 0.905, 0.905], [0.898, 0.898, 0.896], [0.841, 0.840, 0.840],
    #         [0.919, 0.917, 0.917], [0.884, 0.884, 0.884]],
    #     [[0.898, 0.896, 0.898], [0.887, 0.889, 0.891], [0.822, 0.824, 0.824],
    #         [0.903, 0.903, 0.907], [0.850, 0.848, 0.850]],
    #     [[0.892, 0.889, 0.891], [0.885, 0.885, 0.885], [0.813, 0.811, 0.811],
    #         [0.901, 0.901, 0.901], [0.843, 0.838, 0.840]]
    # ])
    # flip_pop_accs = np.mean(flip_pop_accs, -1)

    our_pop_accs = np.array([
        [0.838, 0.827, 0.864, 0.838, 0.824, 0.760, 0.857, 0.903, 0.924, 0.758],
        [0.776, 0.774, 0.757, 0.810, 0.212, 0.552, 0.827, 0.832, 0.887, 0.591],
        [0.720, 0.728, 0.582, 0.765, 0,     0.349, 0.795, 0.667, 0.843, 0.393],
        [0.640, 0.663, 0.478, 0.698, 0,     0.349, 0.704, 0.536, 0.589, 0.356],
        [0.570, 0.582, 0.287, 0.601, 0,     0,     0.716,     0, 0.526, 0.019], 
        [0.467, 0.499,     0, 0.513, 0,     0,     0.626,     0, 0.044,     0],
        [0.409, 0.453,     0, 0.459, 0,     0,     0.557,     0,     0,     0],
        [0.370, 0.422,     0, 0.425, 0,     0,     0.543,     0,     0,     0],
        [0.235, 0.332,     0, 0.333, 0,     0,     0.397,     0,     0,     0]
    ])
    flip_pop_accs = np.array([
        [0.905, 0.898, 0.864, 0.914, 0.861, 0.894, 0.915, 0.974, 0.944, 0.868],
        [0.882, 0.878, 0.818, 0.899, 0.769, 0.832, 0.898, 0.945, 0.899, 0.801],
        [0.875, 0.864, 0.783, 0.880, 0.280, 0.741, 0.880, 0.894, 0.868, 0.725],
        [0.845, 0.845, 0.732, 0.873, 0,     0.587, 0.877, 0.825, 0.834, 0.628],
        [0.810, 0.804, 0.607, 0.840, 0,     0.319, 0.852, 0.584, 0.755, 0.439], 
        [0.744, 0.743, 0.257, 0.762, 0,         0, 0.811, 0.113, 0.582, 0.044],
        [0.702, 0.700, 0.004, 0.720, 0,         0, 0.776,     0, 0.323, 0],
        [0.668, 0.697,     0, 0.684, 0,         0, 0.760,     0, 0.138, 0],
        [0.450, 0.561,     0, 0.527, 0,         0, 0.619,     0,     0, 0]
    ])

    columns = ["Num of Poisons", "Test Acc on Subpop", "Attack"]
    data = []
    # Add data for our attack
    for i in range(our_pop_accs.shape[0]):
        for j in range(our_pop_accs.shape[1]):
            data.append([i, our_pop_accs[i][j], "Our Attack"])

    # Add data for label-flip attack
    for i in range(flip_pop_accs.shape[0]):
        for j in range(flip_pop_accs.shape[1]):
            data.append([i, flip_pop_accs[i][j], "Label-Flip Attack"])

    df = pd.DataFrame(data, columns=columns)

    fig, ax = plt.subplots()
    # sns_plot = sns.boxplot(
    #     x=columns[0], y=columns[1], hue=columns[2], data=df, ax=ax,
    #     medianprops={'linewidth': 3})

    sns_plot = sns.lineplot(
        x=columns[0], y=columns[1], data=df, ax=ax,
        hue=columns[2], style=columns[2],
        markers=True, dashes=False,
        linestyle='', err_style="bars", ci=68
        )
    ax.xaxis.set_ticks(np.arange(len(gen_sizes)))
    ax.xaxis.set_ticklabels(gen_sizes)

    # # Remove legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])

    fig = sns_plot.get_figure()
    fig.savefig("test_on_unseen_models.png")

    # means = np.mean(our_pop_accs, 1)
    # stds = np.std(our_pop_accs, 1)
    # _, _, bars = plt.errorbar(
    #     [str(x) for x in gen_sizes], means, yerr=stds,
    #     fmt='o', label="Our Attack",
    #     elinewidth=4)
    # # [bar.set_alpha(0.75) for bar in bars]

    # means = np.mean(flip_pop_accs, 1)
    # stds = np.std(flip_pop_accs, 1)
    # _, _, bars = plt.errorbar(
    #     [str(x) for x in gen_sizes], means, yerr=stds,
    #     fmt='o', label="Label-Flip Attack",
    #     elinewidth=2)
    # # [bar.set_alpha(0.75) for bar in bars]

    # plt.legend()
    # plt.savefig("test_on_unseen_models.png")
