from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from sklearn.datasets import make_classification
# generate some sthnthetic dataset
generate_dataset = make_classification

import os
import json

import numpy as np
import scipy.sparse as sparse
import scipy.io as sio
import pickle

# from utils import load_adult
# Local running
DATA_FOLDER = 'files/data'
OUTPUT_FOLDER = 'files/results'

def safe_makedirs(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def get_output_mat_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_params.mat' % file_name)

def get_output_dists_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_dists.npz' % file_name)

def get_output_json_path(dataset_name, file_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_defense_results.json' % file_name)

def get_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint, percentile):
    return '%s_attack_clean-centroid_normc-%s_percentile-%s_epsilon-%s.npz' % (dataset_name, norm_sq_constraint, percentile, epsilon)

def get_attack_npz_path(dataset_name, epsilon, norm_sq_constraint, percentile):
    return os.path.join(OUTPUT_FOLDER, 'attack', get_attack_npz_filename(dataset_name, epsilon, norm_sq_constraint, percentile))

def get_target_attack_folder(dataset_name):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name)

def get_target_attack_npz_path(dataset_name, epsilon, weight_decay, percentile, label):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_attack_wd-%s_percentile-%s_epsilon-%s_label-%s.npz' % (dataset_name, weight_decay, percentile, epsilon, label))

def get_target_attack_npz_path_sub(dataset_name, epsilon, sub_ind, weight_decay, percentile, label):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_attack_wd-%s_percentile-%s_epsilon-%s_subind-%s_label-%s.npz' % (dataset_name, weight_decay, percentile, epsilon,sub_ind,label))

def get_attack_results_json_path(dataset_name, weight_decay, percentile, label):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_attack_wd-%s_percentile-%s_label-%s_attackresults.json' % (dataset_name, weight_decay, percentile, label))

def get_attack_results_json_path_sub(dataset_name, weight_decay, percentile, subind, label):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_attack_wd-%s_percentile-%s_subind-%s_label-%s_attackresults.json' % (dataset_name, weight_decay, percentile, subind,label))

def get_timed_results_npz_path(dataset_name, weight_decay, percentile, label):
    return os.path.join(
        OUTPUT_FOLDER,
        dataset_name,
        '%s_attack_wd-%s_percentile-%s_label-%s_timings.npz' % (dataset_name, weight_decay, percentile, label))


def check_orig_data(X_train, Y_train, X_test, Y_test):
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert np.max(Y_train) == 1, 'max of Y_train was %s' % np.max(Y_train)
    assert np.min(Y_train) == -1
    assert len(set(Y_train)) == 2
    assert set(Y_train) == set(Y_test)


def check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified):
    assert X_train.shape[1] == X_poison.shape[1]
    assert X_train.shape[1] == X_modified.shape[1]
    assert X_train.shape[0] + X_poison.shape[0] == X_modified.shape[0]
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_poison.shape[0] == Y_poison.shape[0]
    assert X_modified.shape[0] == Y_modified.shape[0]
    assert X_train.shape[0] * X_poison.shape[0] * X_modified.shape[0] > 0


def load_dogfish():
    dataset_path = os.path.join(DATA_FOLDER)

    train_f = np.load(os.path.join(dataset_path, 'dogfish_900_300_inception_features_train.npz'), allow_pickle = True)
    X_train = train_f['inception_features_val']
    Y_train = np.array(train_f['labels'] * 2 - 1, dtype=int)

    test_f = np.load(os.path.join(dataset_path, 'dogfish_900_300_inception_features_test.npz'))
    X_test = test_f['inception_features_val']
    Y_test = np.array(test_f['labels'] * 2 - 1, dtype=int)

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test


def load_enron_sparse():
    dataset_path = os.path.join(DATA_FOLDER)
    f = np.load(os.path.join(dataset_path, 'enron1_processed_sparse.npz'),allow_pickle = True)

    X_train = f['X_train'].reshape(1)[0]
    Y_train = f['Y_train'] * 2 - 1
    X_test = f['X_test'].reshape(1)[0]
    Y_test = f['Y_test'] * 2 - 1

    assert(sparse.issparse(X_train))
    assert(sparse.issparse(X_test))

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def load_2d_toy(class_sep = 1.0):
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER = 'files/data')
    data_fname = DATA_FOLDER + '/class_sep-{}_2d_toy'.format(class_sep)
    # generate a dataset with 5000 examples, 3000 for train, 2000 for test
    if not os.path.isfile(data_fname):
        full_x, full_y = generate_dataset(n_samples = 5000,
                        n_features=2,
                        n_informative=2,
                        n_redundant=0,
                        n_classes=2,
                        n_clusters_per_class=2,
                        flip_y=0.001,
                        class_sep=class_sep,
                        random_state=0)
        data_full = {}
        data_full['full_x'],data_full['full_y'] = full_x,full_y
        data_file = open(data_fname, 'wb')
        pickle.dump(data_full, data_file,protocol=2)
        data_file.close()
    else:
        data_file = open(data_fname, 'rb')
        f = pickle.load(data_file)
        full_x,full_y = f['full_x'],f['full_y']
    print(full_x[:,1].shape)
    print(full_y.shape)
    print(full_x[:,1].shape)
    print(full_y.shape)

    train_samples = 3000
    # split between train and test datasets
    X_train = full_x[:train_samples]
    X_test = full_x[train_samples:]
    Y_train = full_y[:train_samples]
    Y_test = full_y[train_samples:]
    # convert to {-1,1} as class labels
    Y_train = 2*Y_train-1
    Y_test = 2*Y_test-1
    return X_train, Y_train, X_test, Y_test 

def load_imdb_sparse():
    dataset_path = os.path.join(DATA_FOLDER)
    f = np.load(os.path.join(dataset_path, 'imdb_processed_sparse.npz'), allow_pickle = True)

    X_train = f['X_train'].reshape(1)[0]
    Y_train = f['Y_train'].reshape(-1)
    X_test = f['X_test'].reshape(1)[0]
    Y_test = f['Y_test'].reshape(-1)

    assert(sparse.issparse(X_train))
    assert(sparse.issparse(X_test))

    check_orig_data(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test

def load_adult():
    fname = open(DATA_FOLDER+'/adult_data','rb')
    adult_all = pickle.load(fname)
    X_train = adult_all['X_train']
    Y_train = adult_all['y_train']
    X_test = adult_all['X_test']
    Y_test = adult_all['y_test']
    # print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    return X_train, Y_train, X_test, Y_test 


def load_dataset(dataset_name,class_sep = 1.0):
    if dataset_name == 'imdb':
        return load_imdb_sparse()
    elif dataset_name == 'enron':
        return load_enron_sparse()
    elif dataset_name == 'dogfish':
        return load_dogfish()
    elif dataset_name == 'adult':
        return load_adult()
    elif dataset_name == '2d_toy':
        return load_2d_toy(class_sep=class_sep)
    else:
        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, '%s_train_test.npz' % dataset_name))

        X_train = f['X_train']
        Y_train = f['Y_train'].reshape(-1)
        X_test = f['X_test']
        Y_test = f['Y_test'].reshape(-1)

        check_orig_data(X_train, Y_train, X_test, Y_test)
        return X_train, Y_train, X_test, Y_test


def load_mnist_17():
    return load_dataset('mnist_17')


def load_attack(dataset_name, file_name):
    file_root, ext = os.path.splitext(file_name)

    if ext == '.mat':
        return load_attack_mat(dataset_name, file_name)
    elif ext == '.npz':
        return load_attack_npz(dataset_name, file_name)
    else:
        raise ValueError('File extension must be .mat or .npz.')

def load_attack_mat(dataset_name, file_name, take_path=False):
    if take_path:
        file_path = file_name
    else:
        file_path = os.path.join(OUTPUT_FOLDER, dataset_name, file_name)
    f = sio.loadmat(file_path)

    X_poison = f['X_attack_best']
    Y_poison = f['y_attack_best'].reshape(-1)
    X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)

    if not sparse.issparse(X_train):
        if sparse.issparse(X_poison):
            print('Warning: X_train is not sparse but X_poison is sparse. Densifying X_poison...')
            X_poison = X_poison.toarray()

    for X in [X_train, X_poison, X_test]:
        if sparse.issparse(X): X = X.tocsr()

    if sparse.issparse(X_train):
        X_modified = sparse.vstack((X_train, X_poison), format='csr')
    else:
        X_modified = np.concatenate((X_train, X_poison), axis=0)

    Y_modified = np.concatenate((Y_train, Y_poison), axis=0)

    # Create views into X_modified so that we don't have to keep copies lying around
    num_train = np.shape(X_train)[0]
    idx_train = slice(0, num_train)
    idx_poison = slice(num_train, np.shape(X_modified)[0])
    X_train = X_modified[idx_train, :]
    Y_train = Y_modified[idx_train]
    X_poison = X_modified[idx_poison, :]
    Y_poison = Y_modified[idx_poison]

    check_orig_data(X_train, Y_train, X_test, Y_test)
    check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified)

    return X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison


def load_attack_npz(dataset_name, file_name, take_path=False):
    if take_path:
        file_path = file_name
    else:
        file_path = os.path.join(OUTPUT_FOLDER, dataset_name, file_name)

    f = np.load(file_path)

    if 'X_modified' in f:
        raise AssertionError
        X_modified = f['X_modified']
        Y_modified = f['Y_modified']
        X_test = f['X_test']
        Y_test = f['Y_test']
        idx_train = f['idx_train'].reshape(1)[0]
        idx_poison = f['idx_poison'].reshape(1)[0]
        # Extract sparse array from array wrapper
        if dataset_name in ['enron', 'imdb']:
            X_modified = X_modified.reshape(1)[0]
            X_test = X_test.reshape(1)[0]

        X_train = X_modified[idx_train, :]
        Y_train = Y_modified[idx_train]
        X_poison = X_modified[idx_poison, :]
        Y_poison = Y_modified[idx_poison]

    # Loading KKT attacks, including targeted ones
    elif 'X_poison' in f:
        X_poison = f['X_poison']
        Y_poison = f['Y_poison']
        X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)

        if sparse.issparse(X_train):
            X_poison = X_poison.reshape(1)[0]
            X_modified = sparse.vstack((X_train, X_poison), format='csr')
        else:
            X_modified = np.concatenate((X_train, X_poison), axis=0)

        Y_modified = np.concatenate((Y_train, Y_poison), axis=0)
        idx_train = slice(0, X_train.shape[0])
        idx_poison = slice(X_train.shape[0], X_modified.shape[0])

        if 'idx_to_attack' in f:
            idx_to_attack = f['idx_to_attack']
            X_test = X_test[idx_to_attack, :]
            Y_test = Y_test[idx_to_attack]

        Y_modified = Y_modified.astype(np.float32)
        Y_test = Y_test.astype(np.float32)

    # This is for loading the baselines
    else:
        raise AssertionError
        X_modified = f['poisoned_X_train']
        if dataset_name in ['enron', 'imdb']:
            try:
                X_modified = X_modified.reshape(1)[0]
            except:
                pass

        Y_modified = f['Y_train']

        X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)

        idx_train = slice(0, X_train.shape[0])
        idx_poison = slice(X_train.shape[0], X_modified.shape[0])

        if sparse.issparse(X_modified):
            assert((X_modified[idx_train, :] - X_train).nnz == 0)
        else:
            if sparse.issparse(X_train):
                X_train = X_train.toarray()
                X_test = X_test.toarray()
            assert(np.all(np.isclose(X_modified[idx_train, :], X_train)))
        assert(np.all(Y_modified[idx_train] == Y_train))
        X_poison = X_modified[idx_poison, :]
        Y_poison = Y_modified[idx_poison]

    check_orig_data(X_train, Y_train, X_test, Y_test)
    check_poisoned_data(X_train, Y_train, X_poison, Y_poison, X_modified, Y_modified)

    return X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison
