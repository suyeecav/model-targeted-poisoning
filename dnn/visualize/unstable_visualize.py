import numpy as np
import json
import os
import seaborn as sns
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
plt.rcParams.update({'font.size': 14})


def read_file(x):
    with open(x, 'r') as f:
        data = json.load(f)
    data = np.array(data)
    ppoints, vals = data[:, 1].astype(int), data[:, 2]
    print("Read Tensorboad file!")
    return ppoints, vals


# x, y = read_file(os.path.join(sys.argv[1]))
# # Consider only first 400 points
# # x, y = x[:400], y[:400]

# plt.plot(x, y, color='r', label='Our Attack')
# # columns = ["Num of Poisons", "Test Acc on Subpop"]
# columns = ["Num of Poisons", "Max Loss Diff"]
# plt.xlabel(columns[0])
# plt.ylabel(columns[1])
# # plt.savefig("runs/raw_run_acc.png")
# plt.savefig("runs/raw_run_loss.png")
# exit(0)

# POISON-RATE 0.5
batch_sizes = [32, 64, 128, 256, 512]
poison_rate = 0.4
seed = 3309

accs = [
    [0.9436, 0.8783, 0.9471, 0.9630],
    [0.8307, 0.5802, 0, 0.7637],
    [0.6138, 0.8377, 0, 0.6155],
    [0.8201, 0.8677, 0.7196, 0.7848],
    [0.8977, 0.9206, 0.8924, 0.9383],
]

data = []
columns = ["Batch size", "Test Acc on Subpop"]
for i, acc in enumerate(accs):
    for x in acc:
        data.append([batch_sizes[i], x])

df = pd.DataFrame(data, columns=columns)

fig, ax = plt.subplots()
# sns_plot = sns.boxplot(x=columns[0], y=columns[1], data=df, ax=ax)
sns_plot = sns.stripplot(
    x=columns[0], y=columns[1], data=df, ax=ax, linewidth=1, size=10)

fig = sns_plot.get_figure()
fig.savefig("./runs/varying_bs.png")
print("Saved boxplot!")
exit(0)

seeds_tried = [2021, 24, 105, 418, 69]
batch_size = 128
accs = [
    [0.4356, 0.6578, 0.6367, 0.6737],
    [0.8360, 0.5750, 0.6702, 0.7954],
    [0.8660, 0.6384, 0.7549, 0.7866],
    [0.8113, 0.8818, 0.8571, 0.9312],
    [0.5891, 0.8325, 0.8148, 0.6455],
]

data = []
columns = ["Weight-initialization seed", "Test Acc on Subpop"]
for i, acc in enumerate(accs):
    for x in acc:
        data.append([seeds_tried[i], x])

df = pd.DataFrame(data, columns=columns)

fig, ax = plt.subplots()
# sns_plot = sns.boxplot(x=columns[0], y=columns[1], data=df, ax=ax)
sns_plot = sns.stripplot(
    x=columns[0], y=columns[1], data=df, ax=ax, linewidth=1, size=10)

# Remove legend title
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[0:], labels=labels[0:])

fig = sns_plot.get_figure()
fig.savefig("./runs/varying_seeds.png")
print("Saved boxplot!")
