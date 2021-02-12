import json
import numpy as np
import sys
import os
import torch as ch
import datasets
from tqdm import tqdm
import dnn_utils

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


def get_pop_accs(poison_model_path):
    # Load target model theta_p, set to eval mode
    theta_p = dnn_utils.model_helper("flat")(n_classes=2)
    theta_p = theta_p.cuda()
    theta_p.load_state_dict(ch.load(poison_model_path))
    theta_p.eval()

    # Report performance of poisoned model
    train_loader, test_loader = datasets.dataset_helper("memory")(
        path="./data/datasets/MNIST17/split_1.pt").get_loaders(512)
    # Report accuracy on unseen population data
    (tst_sub_acc, _), _ = dnn_utils.get_model_metrics(
        model=theta_p,
        loader=test_loader,
        target_prop=0)
    return tst_sub_acc


if __name__ == "__main__":
    pick_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    # Plot Tensorboard file
    picked_seed = int(sys.argv[2])
    # columns = ["Num of Poisons", "Test Acc on Subpop"]
    # columns = ["Num of Poisons", "Max Loss Diff"]
    columns = ["Num of Poisons", "Eucledian Distance"]
    # Read JSON file
    x, y = read_file(os.path.join(sys.argv[1], "seed_%d.json" % picked_seed))
    plt.plot(x, y, color='r', label='Our Attack')

    if len(sys.argv) >= 4:
        long_suffix = "mnist17_first/target/arch-flat_target-0_goal-0.05_rule-cycle/loss-ce/"
        # Get experiment runs
        run_data = [[] for _ in range(len(pick_ratios))]
        for mrun in os.listdir(sys.argv[3]):
            rundir = os.path.join(
                sys.argv[3], mrun,
                "seed_%d" % picked_seed,
                long_suffix)

            # Read models inside this
            ratios, accs = [], []
            for mpath in tqdm(os.listdir(rundir)):
                ratios.append(float(mpath.split("ratio-")[1].split("_")[0]))
                accs.append(get_pop_accs(os.path.join(rundir, mpath)))

            accs = np.array(accs)
            accs = accs[np.argsort(ratios)]

            for i in np.argsort(ratios):
                run_data[i].append(accs[i])

        # Compute average
        run_data = np.mean(run_data, 1)

        positions = [int(a * x[-1]) for a in pick_ratios]
        plt.scatter(positions, run_data, color='b', label='Label-Flip Attack')

        plt.legend(loc='lower left')

    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.savefig("plot_seed_%d.png" % picked_seed)
