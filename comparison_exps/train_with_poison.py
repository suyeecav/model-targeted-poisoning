import os
from tqdm import tqdm
from sklearn.svm import LinearSVC
import numpy as np


# Gather poison data from folder
def load_all_data(folder, num_reps=1):
    X, Y, names = [], [], []
    for path in tqdm(os.listdir(folder)):
        data = np.load(os.path.join(folder, path))
        x, y = data['x'], data['y']
        X.append(x)
        Y.append(y)
        names.append(int(path.split(".npz")[0]))
    
    # Order them properly
    ordering = np.argsort(names)
    X, Y = np.array(X), np.array(Y)
    X, Y = X[ordering], Y[ordering]
    
    # Repeat as many times as requested
    X = np.repeat(X, num_reps, axis=0)
    Y = np.repeat(Y, num_reps, axis=0)

    # Onehot to labels
    Y = np.argmax(Y, 1)

    return X, Y


def load_data():
    dataset_path = os.path.join("../files/data/mnist_17_train_test.npz")
    f = np.load(dataset_path)

    x_train = f['X_train']
    y_train = f['Y_train'].reshape(-1)
    x_test = f['X_test']
    y_test = f['Y_test'].reshape(-1)

    oh_train = np.zeros((y_train.shape[0],))
    oh_test = np.zeros((y_test.shape[0],))
    oh_train[y_train == 1] = 1
    oh_test[y_test == 1] = 1
    y_train, y_test = oh_train, oh_test

    return (x_train, y_train), (x_test, y_test), (0, 1)


if __name__ == "__main__":
    import sys
    ratios = [0.05, 0.15, 0.30]

    # Load base data
    (x_train, y_train), (x_test, y_test), (min_val, max_val) = load_data()

    # Load poisoned data
    x_poison, y_poison = load_all_data(sys.argv[1], int(sys.argv[2]))

    for ratio in ratios:

        # Specify how much poison data is to be used at this stage
        cutoff = int(x_train.shape[0] * ratio)

        x_poison_use = x_poison[:cutoff]
        y_poison_use = y_poison[:cutoff]

        print("Using ratio %.2f" % (len(x_poison_use) / len(x_train)))

        # Add poisoned data to main data
        x_use = np.concatenate((x_train, x_poison_use), 0)
        y_use = np.concatenate((y_train, y_poison_use), 0)

        # Define model
        weight_decay=0.09
        C = 1 / (x_use.shape[0] * weight_decay)
        model = LinearSVC(loss='hinge', C=C, tol=1e-10)

        # Train model on poisoned data
        model.fit(x_use, y_use)

        # Log performance for this model
        acc_og_train = model.score(x_train, y_train)
        acc_train = model.score(x_use, y_use)
        acc_test = model.score(x_test, y_test)

        print("Ratio %.2f | Train(D_tr): %.3f , Train(D_tr U D_p) : %.2f | Test : %.3f" % (
            ratio, acc_og_train, acc_og_train, acc_test))
