from secml.ml.classifiers import CClassifierLogistic
from secml.ml.peval.metrics import CMetricAccuracy
from secml.array import CArray

import os
import numpy as np


def load_data():
    dataset_path = os.path.join("../files/data/mnist_17_train_test.npz")
    f = np.load(dataset_path)

    x_train = f['X_train']
    y_train = f['Y_train'].reshape(-1)
    x_test = f['X_test']
    y_test = f['Y_test'].reshape(-1)

    oh_train = np.zeros((y_train.shape[0], 2))
    oh_test = np.zeros((y_test.shape[0], 2))
    oh_train[y_train == -1, 0] = 1
    oh_train[y_train == 1, 1] = 1
    oh_test[y_test == -1, 0] = 1
    oh_test[y_test == 1, 1] = 1
    y_train, y_test = oh_train, oh_test

    y_train = np.argmax(y_train, 1)
    y_test = np.argmax(y_test, 1)

    return (x_train, y_train), (x_test, y_test), (0, 1)


if __name__ == "__main__":
    import sys
    
    # Load main data
    (x_train, y_train), (x_test, y_test), (min_val, max_val) = load_data()

    # Load poison data
    d = np.load(sys.argv[1])
    x_poison, y_poison = d['x'], d['y']

    # Use desired amount of poison data
    usage = [0, 1/6, 1/2, 1]
    how_much_to_use = usage[0]
    if how_much_to_use != 0:
        cutoff = int(len(x_poison) * how_much_to_use)
        x_poison, y_poison = x_poison[:cutoff], y_poison[:cutoff]

        # Combine data
        x_use = np.concatenate((x_train, x_poison), 0)
        y_use = np.concatenate((y_train, y_poison), 0)
    else:
        x_use, y_use = x_train, y_train

    # Convert to CArray
    x_use, y_use = CArray(x_use), CArray(y_use)
    x_test, y_test = CArray(x_test), CArray(y_test)

    print("Poison rato: %.2f" % (len(x_poison) / len(x_train)))

    # Fit classifier
    random_state = 2021

    metric = CMetricAccuracy()
    clf = CClassifierLogistic(C=1)
    clf.fit(x_use, y_use)
    print("Training of classifier complete!")

    # Compute predictions on a test set
    y_pred_tr = clf.predict(x_use)
    y_pred = clf.predict(x_test)

    # Evaluate the accuracy of the original classifier
    tr_acc = metric.performance_score(y_true=y_use, y_pred=y_pred_tr)
    te_acc = metric.performance_score(y_true=y_test, y_pred=y_pred)

    # Report metrics using poisoned model
    print("Poisoned | Train accuracy {:.1%}".format(tr_acc))
    print("Poisoned | Test accuracy {:.1%}".format(te_acc))
