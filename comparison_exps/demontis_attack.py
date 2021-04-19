from secml.ml.classifiers import CClassifierLogistic
from secml.adv.attacks import CAttackPoisoningLogisticRegression
from secml.ml.peval.metrics import CMetricAccuracy
from secml.data import CDataset
from secml.array import CArray

import os
import numpy as np
from sklearn.model_selection import train_test_split


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
    poison_ratio = float(sys.argv[1])
    num_reps = int(sys.argv[2])

    random_state = 2021
    
    # Step 1: Load dataset
    (x_train, y_train), (x_test, y_test), (min_val, max_val) = load_data()

    # Metric to use for training and performance evaluation
    metric = CMetricAccuracy()

    # Creation of the multiclass classifier
    clf = CClassifierLogistic(C=1)

    # Make train-val split from train data for attack purposes
    x_train_red, x_val, y_train_red, y_val = train_test_split(
        x_train, y_train, stratify=y_train, test_size=0.3)
    
    # Convert to CArray
    x_train, y_train = CArray(x_train), CArray(y_train)
    x_test, y_test = CArray(x_test), CArray(y_test)
    x_train_red, y_train_red = CArray(x_train_red), CArray(y_train_red)
    x_val, y_val = CArray(x_val), CArray(y_val)

    # We can now fit the classifier
    clf.fit(x_train_red, y_train_red)
    print("Training of classifier complete!")

    # Compute predictions on a test set
    y_pred = clf.predict(x_test)

    # Should be chosen depending on the optimization problem
    solver_params = {
        'eta': 0.25,
        'eta_min': 2.0,
        'eta_max': None,
        'max_iter': 100,
        'eps': 1e-6
    }

    # Make data wrapper
    tr = CDataset(x_train_red, y_train_red)
    val = CDataset(x_val, y_val)

    pois_attack = CAttackPoisoningLogisticRegression(classifier=clf,
                                      training_data=tr,
                                      val=val,
                                      lb=min_val, ub=max_val,
                                      solver_params=solver_params,
                                      random_seed=random_state)

    # chose and set the initial poisoning sample features and label
    xc = tr[0,:].X
    yc = tr[0,:].Y
    pois_attack.x0 = xc
    pois_attack.xc = xc
    pois_attack.yc = yc

    print("Initial poisoning sample features: {:}".format(xc.ravel()))
    print("Initial poisoning sample label: {:}".format(yc.item()))

    # Number of poisoning points to generate
    n_poisoning_points = int(poison_ratio * y_train.shape[0] / num_reps)
    pois_attack.n_points = n_poisoning_points
    print("Requested %d points with %d reps" % (int(poison_ratio * y_train.shape[0]), num_reps))

    # Run the poisoning attack
    print("Attack started...")
    _, _, pois_ds, _ = pois_attack.run(x_val, y_val)
    print("Attack complete!")

    # Extract, repeat, and save poisoned data
    pr_x, pr_y = pois_ds.X._data._data, pois_ds.Y._data._data
    pr_x = np.repeat(pr_x, num_reps, axis=0)
    pr_y = np.repeat(pr_y, num_reps, axis=0)
    np.savez("LR_data/%.2f_%d_data" % (poison_ratio, num_reps), x=pr_x, y=pr_y)

    print("Adding %d additional points" % len(pr_x))

    # Training of the poisoned classifier
    pois_ds_repeat = CDataset(CArray(pr_x), CArray(pr_y))
    pois_clf = clf.deepcopy()
    # Join the training set with the poisoning points
    pois_tr = tr.append(pois_ds_repeat)
    pois_clf.fit(pois_tr.X, pois_tr.Y)

    # Evaluate the accuracy of the original classifier
    acc = metric.performance_score(y_true=y_test, y_pred=y_pred)

    # Evaluate the accuracy after the poisoning attack
    pois_y_pred = pois_clf.predict(x_test)
    pois_acc = metric.performance_score(y_true=y_test, y_pred=pois_y_pred)

    # Report metrics using poisoned model
    print("Test accuracy on clean model: {:.2%}".format(acc))
    print("Test accuracy on posioned model: {:.2%}".format(pois_acc))
