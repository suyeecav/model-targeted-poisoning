import datasets
import torch as ch
import utils


def stratified_split(D, n_classes=2, split_ratio=0.5):
    X, Y = D.data, D.targets
    split_1, split_2 = [], []
    for i in range(n_classes):
        wanted = ch.nonzero(Y == i).squeeze_(1)
        # Permute for random shuffle
        wanted = wanted[ch.randperm(wanted.shape[0])]
        # Decide split point
        split_point = int(len(wanted) * split_ratio)
        first = wanted[:split_point]
        second = wanted[split_point:]
        split_1.append(first)
        split_2.append(second)

    split_1 = ch.cat(split_1)
    split_2 = ch.cat(split_2)

    data_first = (X[split_1], Y[split_1])
    data_second = (X[split_2], Y[split_2])

    return data_first, data_second


if __name__ == "__main__":
    mnist17 = datasets.dataset_helper("mnist17")()
    train_1, train_2 = stratified_split(mnist17.train)
    val_1, val_2 = stratified_split(mnist17.val)

    # Ensure directory exists
    utils.ensure_dir_exists("./data/datasets/MNIST17/")

    # Save these files
    ch.save({
        "train": {
            "data": train_1[0],
            "targets": train_1[1]
        },
        "val": {
            "data": val_1[0],
            "targets": val_1[1]
        },
    }, "./data/datasets/MNIST17/split_1.pt")

    ch.save({
        "train": {
            "data": train_2[0],
            "targets": train_2[1]
        },
        "val": {
            "data": val_2[0],
            "targets": val_2[1]
        },
    }, "./data/datasets/MNIST17/split_2.pt")

    print("Saved splits!")
