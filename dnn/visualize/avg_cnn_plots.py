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


x, Y = [], []
for i in range(4):
    x, y = read_file(os.path.join(sys.argv[i + 1]))
    Y.append(y)

# Get minimum length across all, cap at that
min_size = min([len(y) for y in Y])
Y = [y[:min_size] for y in Y]
x = x[:min_size]

columns = ["Num of Poisons", "Test Acc on Subpop"]
# columns = ["Num of Poisons", "Max Loss Diff"]
data = []
for i, x_ in enumerate(x):
    for j in range(4):
        data.append([x_, Y[j][i]])
df = pd.DataFrame(data, columns=columns)

sns.set_palette("dark:red_r")
sns_plot = sns.lineplot(
    x=columns[0], y=columns[1], data=df,
    err_style="band", ci=68)
fig = sns_plot.get_figure()
fig.savefig("runs/cnn_avg_acc.png")
# fig.savefig("runs/cnn_avg_loss.png")

# Y = np.mean(Y, 1)
# plt.plot(x, y, color='r', label='Our Attack')
# columns = ["Num of Poisons", "Test Acc on Subpop"]
# # columns = ["Num of Poisons", "Max Loss Diff"]
# plt.xlabel(columns[0])
# plt.ylabel(columns[1])
# plt.savefig("runs/cnn_avg_acc.png")
# plt.savefig("runs/cnn_avg_loss.png")
