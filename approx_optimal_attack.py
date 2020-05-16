from sklearn.datasets import make_classification
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, linear_model
from sklearn import cluster
import csv
import pickle
import sklearn

# import cvxpy as cvx

# KKT attack related modules
import kkt_attack
# from upper_bounds import hinge_loss, hinge_grad, logistic_grad
from datasets import load_dataset

import data_utils as data
import argparse
import os
import sys

from sklearn.externals import joblib

# import adaptive attack related functions
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',default='svm',help='victim model type: SVM or rlogistic regression')
# ol: target classifier is from the adapttive attack, kkt: target is from kkt attack, real: actual classifier, compare: compare performance
# of kkt attack and adaptive attack using same stop criteria
parser.add_argument('--target_model', default='all',help='set your target classifier, options: kkt, ol, real, all')
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, synthetic")
parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')

# some params related to online algorithm, use the default
parser.add_argument('--online_alg_criteria',default='max_loss',help='stop criteria of online alg: max_loss, norm, trn_acc_to_tar')
# parser.add_argument('--incre_tol_par',default=1e-2,help='stop value of online alg: max_loss or norm')
parser.add_argument('--weight_decay',default=0.09,help='weight decay for regularizers')
parser.add_argument('--incre_tol_par',default=1e-2,help='stop value of online alg: max_loss or norm')

args = parser.parse_args()

####################### set up the poisoning attack parameters #######################################

# KKT attack specific parameters
percentile = 90
loss_percentile = 90
use_slab = False
use_loss = False
use_l2 = False
dataset_name = args.dataset

if args.target_model == "all":
    target_model_names = ['real','ol','kkt']
elif args.target_model == "kkt":
    target_model_names = ['kkt']
elif args.target_model == "ol":
    target_model_names = ['ol']
elif args.target_model == "real":
    target_model_names = ['real']
else:
    print("please provide a valid target classifier!")
    sys.exit(0)

assert dataset_name in ['adult','mnist_17']
if dataset_name == 'mnist_17':
    args.poison_whole = True
    # args.incre_tol_par = 0.1

if args.model_type == 'svm':
    print("chosen model: svm")
    ScikitModel = svm_model
    model_grad = hinge_grad
else:
    print("chosen model: lr")
    ScikitModel = logistic_model
    model_grad = logistic_grad

# norm_sq_constraint = 1.0

learning_rate = 0.01
online_alg = "incremental"

######################################################################

################# Main body of work ###################
# creat files that store clustering info
make_dirs(args.dataset)

no_save_files = True
if not no_save_files:
    if args.target_model == "all":
        kkt_approx_optimal_attack_file = open('files/results/{}/approx_optimal_attack/kkt_approx_ideal_attack_{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
        kkt_approx_optimal_attack_writer = csv.writer(kkt_approx_optimal_attack_file, delimiter=str(' ')) 

        real_approx_optimal_attack_file = open('files/results/{}/approx_optimal_attack/real_approx_ideal_attack_{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
        real_approx_optimal_attack_writer = csv.writer(real_approx_optimal_attack_file, delimiter=str(' ')) 

        ol_approx_optimal_attack_file = open('files/results/{}/approx_optimal_attack/ol_approx_ideal_attack_{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
        ol_approx_optimal_attack_writer = csv.writer(ol_approx_optimal_attack_file, delimiter=str(' ')) 
    elif args.target_model == "kkt":
        kkt_approx_optimal_attack_file = open('files/results/{}/approx_optimal_attack/kkt_approx_ideal_attack_{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
        kkt_approx_optimal_attack_writer = csv.writer(kkt_approx_optimal_attack_file, delimiter=str(' ')) 
    elif args.target_model == "real":
        real_approx_optimal_attack_file = open('files/results/{}/approx_optimal_attack/real_approx_ideal_attack_{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
        real_approx_optimal_attack_writer = csv.writer(real_approx_optimal_attack_file, delimiter=str(' ')) 
    elif args.target_model == "ol":
        ol_approx_optimal_attack_file = open('files/results/{}/approx_optimal_attack/ol_approx_ideal_attack_{}.csv'.format(args.dataset,args.incre_tol_par), 'w')
        ol_approx_optimal_attack_writer = csv.writer(ol_approx_optimal_attack_file, delimiter=str(' ')) 

# load data
X_train, y_train, X_test, y_test = load_dataset(args.dataset)

if min(y_test)>-1:
    y_test = 2*y_test-1
if min(y_train) > -1:
    y_train = 2*y_train - 1

full_x = np.concatenate((X_train,X_test),axis=0)
full_y = np.concatenate((y_train,y_test),axis=0)
if args.dataset == "synthetic":
    # get the min and max value of features
    # if constrained:
    x_pos_min, x_pos_max = np.amin(full_x[full_y == 1]),np.amax(full_x[full_y == 1])
    x_neg_min, x_neg_max = np.amin(full_x[full_y == -1]),np.amax(full_x[full_y == -1])
    x_pos_tuple = (x_pos_min,x_pos_max)
    x_neg_tuple = (x_neg_min,x_neg_max)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
    print("max values of the features of synthetic dataset:")
    print(x_pos_min,x_pos_max,x_neg_min,x_neg_max)
elif args.dataset in ["adult","mnist_17"]:
    x_pos_tuple = (0,1)
    x_neg_tuple = (0,1)
    x_lim_tuples = [x_pos_tuple,x_neg_tuple]
else:
    x_pos_tuple = None
    x_neg_tuple = None

# data preprocessers for the current data
class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
X_train,
y_train,
percentile=percentile)

# do clustering and test if these fit previous clusters
# do clustering and test if these fit previous clusters
if args.poison_whole:
    cl_inds, cl_cts = [0], [0]
else:
    cls_fname = 'files/data/{}_trn_cluster_labels.txt'.format(args.dataset)
    if os.path.isfile(cls_fname):
        trn_km = np.loadtxt(cls_fname)
        cls_fname = 'files/data/{}_tst_cluster_labels.txt'.format(args.dataset)
        tst_km = np.loadtxt(cls_fname)
    else:
        print("please first generate the target classifier and obtain subpop info!")
        sys.exit(1) 
    # find the selected clusters and corresponding subpop size
    # cl_inds, cl_cts = np.unique(trn_km, return_counts=True)
    cls_fname = 'files/data/{}_selected_subpops.txt'.format(dataset_name)
    selected_subpops = np.loadtxt(cls_fname)
    cl_inds = selected_subpops[0]
    cl_cts = selected_subpops[1]

if dataset_name == "adult":
    pois_rates = [0.05]
elif dataset_name == "mnist_17":
    pois_rates = [0.2]

# search step size for kkt attack
epsilon_increment = 0.005
# train unpoisoned model
C = 1.0 / (X_train.shape[0] * args.weight_decay)
fit_intercept = True
model = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=24,
            verbose=False,
            max_iter = 1000)
model.fit(X_train, y_train)

# use this to check ideal classifier performance for KKT attack
model_dumb = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=24,
            verbose=False,
            max_iter = 1000)
model_dumb.fit(X_train, y_train)

model_dumb1 = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=24,
            verbose=False,
            max_iter = 1000)
model_dumb1.fit(X_train[0:2000], y_train[0:2000])

kkt_model_p = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=24,
            verbose=False,
            max_iter = 1000)
kkt_model_p.fit(X_train, y_train)

# report performance of clean model
clean_acc = model.score(X_test,y_test)
print("Clean Total Acc:",clean_acc)
margins = y_train*(X_train.dot(model.coef_.reshape(-1)) + model.intercept_)
clean_total_loss = np.sum(np.maximum(1-margins, 0))
print("clean model loss on train:",clean_total_loss)
# print("clean model theta and bias:",model.coef_,model.intercept_)

X_train_cp, y_train_cp = np.copy(X_train), np.copy(y_train)
for cl_ind in cl_inds:
    cl_ind = int(cl_ind)
    if args.poison_whole:
        tst_sub_x, tst_sub_y = X_test, y_test 
        tst_nsub_x, tst_nsub_y = X_test,y_test
        trn_sub_x, trn_sub_y = X_train, y_train
        trn_nsub_x, trn_nsub_y = X_train, y_train
    else:
        tst_sbcl = np.where(np.logical_and(tst_km==cl_ind,y_test == -1))
        trn_sbcl = np.where(np.logical_and(trn_km==cl_ind,y_train == -1))
        tst_non_sbcl = np.where(np.logical_or(tst_km!=cl_ind,y_test != -1))
        trn_non_sbcl = np.where(np.logical_or(trn_km!=cl_ind,y_train != -1))
        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y = X_train_cp[trn_sbcl], y_train_cp[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train_cp[trn_non_sbcl], y_train_cp[trn_non_sbcl]
    
        # make sure subpop is from class -1
        assert (tst_sub_y == -1).all()
        assert (trn_sub_y == -1).all()

    print("----------Subpop Indx: {}------".format(cl_ind))
    print('Clean Overall Test Acc : %.3f' % model.score(X_test, y_test))
    print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
    print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
    print('Clean Test Target Acc : %.3f' % model.score(tst_sub_x, tst_sub_y))
    print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))
    sub_frac = 1
    # store the approximate optimal attack info
    for target_model_name in target_model_names:
        approx_optimal_attacks = []
        for total_epsilon in pois_rates:
            print("------ Attacking target model {} with KKT fixed Ratio of {} ----".format(target_model_name,total_epsilon))
            epsilon_pairs = []
            min_stop_criteria = 1e10
            # load the target model designed for kkt model
            if target_model_name == 'real':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/whole-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/subpop-{}_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                data_info = np.load(filename)
                best_target_theta = data_info["best_target_theta"]
                best_target_bias = data_info["best_target_bias"]
            elif target_model_name == 'ol':
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                target_model_p_online = joblib.load(open(filename, 'rb'))
                best_target_theta = target_model_p_online.coef_.reshape(-1)
                best_target_bias = target_model_p_online.intercept_
                best_target_bias = best_target_bias[0]
            elif target_model_name == 'kkt':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/whole-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/subpop-{}_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                target_kkt_model_p = joblib.load(open(filename, 'rb'))
                best_target_theta = target_kkt_model_p.coef_.reshape(-1)
                best_target_bias = target_kkt_model_p.intercept_
                best_target_bias = best_target_bias[0]

            # instantiate the "target" classifier
            model_dumb1.coef_ = np.array([best_target_theta])
            model_dumb1.intercept_ = np.array([best_target_bias]) 

            margins = trn_sub_y*(trn_sub_x.dot(best_target_theta) + best_target_bias)
            _, ideal_target_err = calculate_loss(margins)
            print("Ideal Target Train Acc:",1-ideal_target_err)
            ideal_tar_acc= 1-ideal_target_err
            margins =trn_nsub_y*(trn_nsub_x.dot(best_target_theta) + best_target_bias)
            _, ideal_collat_err = calculate_loss(margins)
            print("Ideal Collat Train Acc:",1-ideal_collat_err)
            margins = y_train*(X_train.dot(best_target_theta) + best_target_bias)
            _, ideal_total_err = calculate_loss(margins)
            print("Ideal Total Train Acc:",1-ideal_total_err)

            margins = tst_sub_y*(tst_sub_x.dot(best_target_theta) + best_target_bias)
            _, ideal_target_err = calculate_loss(margins)
            print("Ideal Target Test Acc:",1-ideal_target_err)
            margins =tst_nsub_y*(tst_nsub_x.dot(best_target_theta) + best_target_bias)
            _, ideal_collat_err = calculate_loss(margins)
            print("Ideal Collat Test Acc:",1-ideal_collat_err)
            margins = y_test*(X_test.dot(best_target_theta) + best_target_bias)
            _, ideal_total_err = calculate_loss(margins)
            print("Ideal Total Test Acc:",1-ideal_total_err)

            # load the lower bound on number of poisoned points required
            if target_model_name == 'real':
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            elif target_model_name == 'kkt':
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)   
            elif target_model_name == 'ol':
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par) 

            data_info = np.load(filename) 
            best_lower_bound = data_info["best_lower_bound"]
            best_max_loss_x = data_info["best_max_loss_x"]
            best_max_loss_y = data_info["best_max_loss_y"]
            # check if we can reproduce the results with using the max_loss points
            X_tar = np.repeat(best_max_loss_x, best_lower_bound, axis=0)
            Y_tar = np.repeat(best_max_loss_y, best_lower_bound, axis=0)
            X_train_max_loss = np.concatenate((X_train,X_tar),axis = 0)
            y_train_max_loss = np.concatenate((y_train,Y_tar),axis = 0)
            print("Shape of newly added max loss points",X_train_max_loss.shape, y_train_max_loss.shape)

            C = 1.0 / (X_train_max_loss.shape[0] * args.weight_decay)
            model_p_ol = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
            model_p_ol.fit(X_train_max_loss, y_train_max_loss)                 
            # acc on subpop and rest of pops
            trn_target_acc = model_p_ol.score(trn_sub_x, trn_sub_y)
            print()
            print('(Max Loss) Train Total Acc : ', model_p_ol.score(X_train, y_train))
            print('(Max Loss) Train Target Acc : ', trn_target_acc)
            print('(Max Loss) Train Collat Acc : ', model_p_ol.score(trn_nsub_x,trn_nsub_y))

            print('(Max Loss) Test Total Acc : ', model_p_ol.score(X_test, y_test))
            print('(Max Loss) Test Target Acc : ', model_p_ol.score(tst_sub_x, tst_sub_y))
            print('(Max Loss) Test Collat Acc : ', model_p_ol.score(tst_nsub_x, tst_nsub_y))            
            sys.exit(0)

            exact_epsilon = float(best_lower_bound)/X_train.shape[0] # use this as the ratio for kkt attack
            print("theoretical lower bound for achieving the classifier is:",best_lower_bound)
            print("poison ratio for approximate optimal attack:",exact_epsilon)
            # start the process of generating the approximate optimal attack with kkt attack
            two_class_kkt, clean_grad_at_target_theta, target_bias_grad, max_losses = kkt_attack.kkt_setup(
                best_target_theta,
                best_target_bias,
                X_train_cp, y_train_cp,
                X_test, y_test,
                dataset_name,
                percentile,
                loss_percentile,
                model_dumb,
                model_grad,
                class_map,
                use_slab,
                use_loss,
                use_l2,
                x_pos_tuple=x_pos_tuple,
                x_neg_tuple=x_neg_tuple)

            target_grad = clean_grad_at_target_theta + ((1 + exact_epsilon) * args.weight_decay * best_target_theta)
            epsilon_neg = (exact_epsilon - target_bias_grad) / 2
            epsilon_pos = exact_epsilon - epsilon_neg

            if (epsilon_neg >= 0) and (epsilon_neg <= exact_epsilon):
                epsilon_pairs.append((epsilon_pos, epsilon_neg))

            for epsilon_pos in np.arange(0, exact_epsilon + 1e-6, epsilon_increment):
                epsilon_neg = exact_epsilon - epsilon_pos
                epsilon_pairs.append((epsilon_pos, epsilon_neg))
            
            for epsilon_pos, epsilon_neg in epsilon_pairs:
                print('\n## Trying epsilon_pos %s, epsilon_neg %s' % (epsilon_pos, epsilon_neg))
                X_modified, Y_modified, obj, x_pos, x, num_pos, num_neg = kkt_attack.kkt_attack(
                    two_class_kkt,
                    target_grad, best_target_theta,
                    exact_epsilon * sub_frac, epsilon_pos * sub_frac, epsilon_neg * sub_frac,
                    X_train_cp, y_train_cp,
                    class_map, centroids, centroid_vec, sphere_radii, slab_radii,
                    best_target_bias, target_bias_grad, max_losses)
                
                # separate out the poisoned points
                idx_poison = slice(X_train.shape[0], X_modified.shape[0])
                idx_clean = slice(0, X_train.shape[0])
                print("Shape of poisoned data and clean data:",X_modified.shape,X_train.shape)
                X_poison = X_modified[idx_poison,:]
                Y_poison = Y_modified[idx_poison]   
                # unique points and labels in kkt attack
                unique_x, unique_indices, unique_counts = np.unique(X_poison,return_index = True,return_counts = True,axis=0)
                unique_y = Y_poison[unique_indices]
                # print("sanity check: shape of unique_x:",unique_x.shape)                
                # retrain the model 
                C = 1.0 / (X_modified.shape[0] * args.weight_decay)
                model_p = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 1000)
                model_p.fit(X_modified, Y_modified)                 
                # acc on subpop and rest of pops
                trn_target_acc = model_p.score(trn_sub_x, trn_sub_y)
                print()
                print('Train Total Acc : ', model_p.score(X_train, y_train))
                print('Train Target Acc : ', trn_target_acc)
                print('Train Collat Acc : ', model_p.score(trn_nsub_x,trn_nsub_y))

                print('Test Total Acc : ', model_p.score(X_test, y_test))
                print('Test Target Acc : ', model_p.score(tst_sub_x, tst_sub_y))
                print('Test Collat Acc : ', model_p.score(tst_nsub_x, tst_nsub_y))

                # sanity check on the max loss difference between target model and kkt model
                kkt_tol_par_max_loss = -1
                for y_b in set(y_train):
                    if y_b == 1:
                        max_loss_diff,_ = search_max_loss_pt(model_p,model_dumb1,y_b,x_pos_tuple,args)
                        if kkt_tol_par_max_loss < max_loss_diff:
                            kkt_tol_par_max_loss = max_loss_diff
                    elif y_b == -1:
                        max_loss_diff,_ = search_max_loss_pt(model_p,model_dumb1,y_b,x_neg_tuple,args)
                        if kkt_tol_par_max_loss < max_loss_diff:
                            kkt_tol_par_max_loss = max_loss_diff
                print("max loss difference between target and kkt model is:",kkt_tol_par_max_loss)
                kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-model_p.coef_)**2+(model_dumb1.intercept_.reshape(-1) - model_p.intercept_)**2)[0]
                print("norm difference between target and kkt model is:",kkt_tol_par_norm)
                if args.online_alg_criteria == "max_loss":
                    kkt_tol_par = kkt_tol_par_max_loss
                elif args.online_alg_criteria == "norm":
                    kkt_tol_par = kkt_tol_par_norm
                elif args.online_alg_criteria == "trn_acc_to_tar":
                    # choose the one with lowest abs distance to the target model
                    kkt_tol_par = np.abs(ideal_tar_acc - trn_target_acc) 

                if kkt_tol_par < min_stop_criteria:
                    min_stop_criteria = kkt_tol_par
                    # store the best kkt model params and data
                    kkt_model_p.coef_ = np.copy(model_p.coef_)
                    kkt_model_p.intercept_ = np.copy(model_p.intercept_)
                    kkt_x_modified = np.copy(X_modified)
                    kkt_y_modified = np.copy(Y_modified)

            best_kkt_theta = kkt_model_p.coef_.reshape(-1)
            best_kkt_bias = kkt_model_p.intercept_
            best_kkt_bias = best_kkt_bias[0]

            idx_poison = slice(X_train.shape[0], kkt_x_modified.shape[0])
            idx_clean = slice(0, X_train.shape[0])
            
            X_poison = kkt_x_modified[idx_poison,:]
            Y_poison = kkt_y_modified[idx_poison]  

            kkt_unique_x, kkt_unique_indices, kkt_unique_counts = np.unique(X_poison,return_index = True,return_counts = True,axis=0)
            kkt_unique_y = Y_poison[kkt_unique_indices]

            # save the model weights and the train and test data
            if target_model_name == 'real':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/whole-{}_approx_optimal_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/subpop-{}_approx_optimal_for_real_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            elif target_model_name == 'kkt':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/whole-{}_approx_optimal_for_kkt__model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/subpop-{}_approx_optimal_for_kkt_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            elif target_model_name == 'ol':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/whole-{}_approx_optimal_for_online_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/subpop-{}_approx_optimal_for_online_model_tol-{}.sav'.format(dataset_name,cl_ind,args.incre_tol_par)
            joblib.dump(kkt_model_p, filename)

            if target_model_name == 'real':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/whole-{}_approx_optimal_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/subpop-{}_approx_optimal_for_real_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            elif target_model_name == 'kkt':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/whole-{}_approx_optimal_for_kkt__data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/subpop-{}_approx_optimal_for_kkt_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
            elif target_model_name == 'ol':
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/whole-{}_approx_optimal_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)
                else:
                    filename = 'files/kkt_models/{}/approx_optimal_attack/subpop-{}_approx_optimal_for_online_data_tol-{}.npz'.format(dataset_name,cl_ind,args.incre_tol_par)

            np.savez(filename,
                    kkt_x_modified = kkt_x_modified,
                    kkt_y_modified = kkt_y_modified,
                    kkt_unique_x = kkt_unique_x,
                    kkt_unique_y = kkt_unique_y,
                    kkt_unique_counts = kkt_unique_counts,
                    best_target_theta = best_target_theta,
                    best_target_bias = best_target_bias,
                    best_kkt_theta = best_kkt_theta,
                    best_kkt_bias = best_kkt_bias,
                    best_max_loss_x = best_max_loss_x,
                    best_max_loss_y = best_max_loss_y
                    )

            # sanity check on the max loss difference between target model and kkt model
            print("----- some info of the selected kkt model and its target model ---")
            print('Train Total Acc : ', kkt_model_p.score(X_train, y_train))
            print('Test Total Acc : ', kkt_model_p.score(X_test, y_test))
            print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
            print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))
            print('Test Target Acc : ', kkt_model_p.score(tst_sub_x,tst_sub_y))
            print('Test Collat Acc : ', kkt_model_p.score(tst_nsub_x,tst_nsub_y))
            model_dumb1.coef_ = np.array([best_target_theta])
            model_dumb1.intercept = np.array([best_target_bias])
            kkt_tol_par_max_loss = -1
            for y_b in set(y_train):
                if y_b == 1:
                    max_loss_diff,_ = search_max_loss_pt(kkt_model_p,model_dumb1,y_b,x_pos_tuple,args)
                    if kkt_tol_par_max_loss < max_loss_diff:
                        kkt_tol_par_max_loss = max_loss_diff
                elif y_b == -1:
                    max_loss_diff,_ = search_max_loss_pt(kkt_model_p,model_dumb1,y_b,x_neg_tuple,args)
                    if kkt_tol_par_max_loss < max_loss_diff:
                        kkt_tol_par_max_loss = max_loss_diff
            print("max loss difference between selected target and selected kkt model is:",kkt_tol_par_max_loss)
            kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-kkt_model_p.coef_)**2+(model_dumb1.intercept_.reshape(-1) - kkt_model_p.intercept_)**2)
            print("norm difference between selected target and selected kkt model is:",kkt_tol_par_norm)
            print("minimum of the stop criteria:",min_stop_criteria)
            approx_optimal_attacks = approx_optimal_attacks + [best_lower_bound,kkt_tol_par_max_loss,kkt_tol_par_norm]
        if target_model_name == 'kkt':
            kkt_approx_optimal_attack_writer.writerow(approx_optimal_attacks)
        elif target_model_name == 'real':
            real_approx_optimal_attack_writer.writerow(approx_optimal_attacks)
        elif target_model_name == 'ol':
            ol_approx_optimal_attack_writer.writerow(approx_optimal_attacks)

# close all files
if args.target_model == "all":
    kkt_approx_optimal_attack_file.flush()
    kkt_approx_optimal_attack_file.close()

    real_approx_optimal_attack_file.flush()
    real_approx_optimal_attack_file.close()

    ol_approx_optimal_attack_file.flush()
    ol_approx_optimal_attack_file.close()
elif args.target_model == "kkt":
    kkt_approx_optimal_attack_file.flush()
    kkt_approx_optimal_attack_file.close()
elif args.target_model == "real":
    real_approx_optimal_attack_file.flush()
    real_approx_optimal_attack_file.close()
elif args.target_model == "ol":
    ol_approx_optimal_attack_file.flush()
    ol_approx_optimal_attack_file.close()