from art.attacks.poisoning import PoisoningAttackSVM
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from art.estimators.classification import SklearnClassifier
from torch.utils.tensorboard import SummaryWriter
from art.utils import load_mnist
import os
from tqdm import tqdm
import argparse


def get_acc(clf, X, Y):
    predictions = clf.predict(X)
    accuracy = np.sum(np.argmax(predictions, axis=1) ==
                      np.argmax(Y, axis=1)) / len(Y)
    return accuracy


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

    return (x_train, y_train), (x_test, y_test), (0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_copies', default=1,
                        type=int, help='Number of copies per poison-point')
    parser.add_argument('--ratio', default=0.3,
                        type=float, help='Poison ratio')
    parser.add_argument('--data_path',
                        help='Folder to save datapoints in')
    parser.add_argument('--log_path',
                        help='Folder to log information in')
    args = parser.parse_args()

    # Step 0: Misc setups
    weight_decay = 0.09
    tol = 1e-8
    writer = SummaryWriter(args.log_path)

    # Step 1: Load dataset
    (x_train, y_train), (x_test, y_test), (min_val, max_val) = load_data()

    # Make train-val split from train data
    # For attack purposes
    x_train_red, x_val, y_train_red, y_val = train_test_split(
        x_train, y_train, stratify=y_train, test_size=0.3)

    # Step 2: Create the model
    C = 1 / (x_train_red.shape[0] * weight_decay)
    model = LinearSVC(loss='hinge', C=C, tol=tol)

    # Step 3: Create the ART classifier
    classifier = SklearnClassifier(
        model=model, clip_values=(min_val, max_val))

    # Step 4: Train the ART classifier
    classifier.fit(x_train, y_train)

    # Step 5: Evaluate the ART classifier on benign test examples
    accuracy = get_acc(classifier, x_test, y_test)
    print("Test accuracy on clean model: %.3f" % (accuracy * 100))

    # Step 6: Generate adversarial test examples
    n_elems = x_train.shape[0]
    n_poisons = int(args.ratio * n_elems) // args.n_copies
    seeds = np.random.permutation(n_elems)[: n_poisons]
    x_seed, y_seed = x_train[seeds], y_train[seeds]
    # Flip seed labels
    y_seed = 1 - y_seed
    x_poison, y_poison = [], []
    iterator = tqdm(range(n_poisons))

    for i in iterator:
        # Generate poison point
        attack = PoisoningAttackSVM(
            classifier=classifier,
            step=0.01, eps=0.2,
            x_train=x_train_red, y_train=y_train_red,
            x_val=x_val, y_val=y_val,
            max_iter=100)
        point = attack.generate_attack_point(x_seed[i], y_seed[i])
        xp_i, yp_i = point[0], y_seed[i]
        
        # Save poison point
        np.savez(os.path.join(args.data_path, str(i)), x=xp_i, y=yp_i)

        # Add poison point
        for _ in range(args.n_copies):
            x_poison.append(xp_i)
            y_poison.append(yp_i)
        x_use = np.concatenate((x_train_red, x_poison), 0)
        y_use = np.concatenate((y_train_red, y_poison), 0)

        # Retrain SVM
        C = 1 / (x_use.shape[0] * weight_decay)
        model = LinearSVC(loss='hinge', C=C, tol=tol)
        classifier = SklearnClassifier(
            model=model, clip_values=(min_val, max_val))
        classifier.fit(x_use, y_use)

        # Compute model performance on various data
        accuracy_unseen = get_acc(classifier, x_test, y_test)
        accuracy_seen = get_acc(classifier, x_train, y_train)
        accuracy_seen_og = get_acc(classifier, x_use, y_use)
        accuracy_train_split = get_acc(classifier, x_train_red, y_train_red)
        accuracy_val_split = get_acc(classifier, x_val, y_val)

        # Log informaton
        iterator.set_description(
            "Accuracy so far: %.3f | Poison points added: %d" % (accuracy, len(y_poison)))
        writer.add_scalar('Accuracy (train-og)', accuracy_seen_og, (i + 1) * args.n_copies)
        writer.add_scalar('Accuracy (train-split)', accuracy_train_split, (i + 1) * args.n_copies)
        writer.add_scalar('Accuracy (val-split)', accuracy_val_split, (i + 1) * args.n_copies)
        writer.add_scalar('Accuracy (train-poisoned)', accuracy_seen, (i + 1) * args.n_copies)
        writer.add_scalar('Accuracy (val)', accuracy_unseen, (i + 1) * args.n_copies)
        

    # Close open writer
    writer.close()
