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
parser.add_argument('--target_model', default='real',help='set your target classifier, options: just keep it as real')
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, synthetic")
parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')

# some params related to online algorithm, use the default
parser.add_argument('--online_alg_criteria',default='max_loss',help='stop criteria of online alg: max_loss or norm')
parser.add_argument('--incre_tol_par',default=1e-2,help='stop value of online alg: max_loss or norm')
parser.add_argument('--weight_decay',default=0.09,help='weight decay for regularizers')

args = parser.parse_args()

####################### set up the poisoning attack parameters #######################################

# KKT attack specific parameters
percentile = 90
loss_percentile = 90
use_slab = False
use_loss = False
use_l2 = False
dataset_name = args.dataset

assert dataset_name in ['adult','mnist_17']
if dataset_name == 'mnist_17':
    args.poison_whole = True
    # see if decreasing by half helps
    args.incre_tol_par = 0.05
    args.weight_decay = 0.09
elif dataset_name == 'adult':
    args.weight_decay = 1e-5
    args.incre_tol_par = 1e-3


if args.model_type == 'svm':
    print("chosen model: svm")
    ScikitModel = svm_model
    model_grad = hinge_grad
else:
    print("chosen model: lr")
    ScikitModel = logistic_model
    model_grad = logistic_grad

# norm_sq_constraint = 1.0
max_loss_tol_par = 1e-2

learning_rate = 0.01
online_alg = "incremental"

######################################################################

################# Main body of work ###################
# creat files that store clustering info
make_dirs(args.dataset)
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

for kk in range(len(cl_inds)):
    cl_ind = int(cl_inds[kk])
    if args.poison_whole:
        check_valid_thetas_file = open('files/results/{}/check_valid_thetas/whole-{}_check_valid_thetas_tol-{}.csv'.format(\
            args.dataset,cl_ind,args.incre_tol_par), 'w')
    else:
        check_valid_thetas_file = open('files/results/{}/check_valid_thetas/subpop-{}_check_valid_thetas_tol-{}.csv'.format(\
            args.dataset,cl_ind,args.incre_tol_par), 'w')

    check_valid_thetas_writer = csv.writer(check_valid_thetas_file, delimiter=str(' ')) 

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

    subpop_data = [trn_sub_x,trn_sub_y,trn_nsub_x,trn_nsub_y,\
        tst_sub_x,tst_sub_y,tst_nsub_x,tst_nsub_y]

    test_target = model.score(tst_sub_x, tst_sub_y)
    test_collat = model.score(tst_nsub_x, tst_nsub_y)

    print("----------Subpop Indx: {}------".format(cl_ind))
    print('Clean Overall Test Acc : %.3f' % model.score(X_test, y_test))
    print('Clean Test Target Acc : %.3f' % test_target)
    print('Clean Test Collat Acc : %.3f' % test_collat)
    print('Clean Overall Train Acc : %.3f' % model.score(X_train, y_train))
    print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
    print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))

    orig_model_acc_scores = []
    orig_model_acc_scores.append(model.score(X_test, y_test))
    orig_model_acc_scores.append(test_target)
    orig_model_acc_scores.append(test_collat)
    orig_model_acc_scores.append(model.score(X_train, y_train))
    orig_model_acc_scores.append(model.score(trn_sub_x, trn_sub_y))
    orig_model_acc_scores.append(model.score(trn_nsub_x,trn_nsub_y))
    # load target classifiers
    if args.poison_whole:
        fname = open('files/target_classifiers/{}/opt_thetas_whole'.format(dataset_name), 'rb')  
        
    else:
        fname = open('files/target_classifiers/{}/opt_thetas_subpop_{}'.format(dataset_name,cl_ind), 'rb')  
    f = pickle.load(fname)

    target_thetas = f['valid_thetas']
    target_biases = f['valid_biases']
    sub_frac = 1

    ## apply online learning algorithm to provide lower bound and candidate attack ##
    C = 1.0 / (X_train.shape[0] * args.weight_decay)
    curr_model = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
    curr_model.fit(X_train, y_train)

    target_model = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=fit_intercept,
                random_state=24,
                verbose=False,
                max_iter = 1000)
    target_model.fit(X_train, y_train)
    # test each of the valid thetas separately
    for i in range(len(target_thetas)):
        print("************** Target Classifier {} for Subpop:{} ***************".format(i,cl_ind))
        # if not fit_intercept:
        #     target_bias = 0
        best_target_theta = target_thetas[i]
        best_target_bias = target_biases[i]

        # # print info of the target classifier and also store these info# #
        print("--- Acc Info of Actual Target Classifier ---")
        target_model_acc_scores = []
        margins = tst_sub_y*(tst_sub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_target_err = calculate_loss(margins)
        margins =tst_nsub_y*(tst_nsub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_collat_err = calculate_loss(margins)
        margins =y_test*(X_test.dot(best_target_theta) + best_target_bias)
        _, ideal_total_err = calculate_loss(margins)
        print("Ideal Total Test Acc:",1-ideal_total_err)
        print("Ideal Target Test Acc:",1-ideal_target_err)
        print("Ideal Collat Test Acc:",1-ideal_collat_err)
        target_model_acc_scores.append(1-ideal_total_err)
        target_model_acc_scores.append(1-ideal_target_err)
        target_model_acc_scores.append(1-ideal_collat_err)

        margins = trn_sub_y*(trn_sub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_target_err = calculate_loss(margins)
        margins =trn_nsub_y*(trn_nsub_x.dot(best_target_theta) + best_target_bias)
        _, ideal_collat_err = calculate_loss(margins)
        margins =y_train*(X_train.dot(best_target_theta) + best_target_bias)
        ideal_total_loss, ideal_total_err = calculate_loss(margins)
        print("Ideal Total Train Acc:",1-ideal_total_err)
        print("Ideal Target Train Acc:",1-ideal_target_err)
        print("Ideal Collat Train Acc:",1-ideal_collat_err)
        target_model_acc_scores.append(1-ideal_total_err)
        target_model_acc_scores.append(1-ideal_target_err)
        target_model_acc_scores.append(1-ideal_collat_err)

        # the regularized total loss on clean dataset
        loss_Dc = X_train.shape[0] * (ideal_total_loss + (args.weight_decay/2) * np.linalg.norm(best_target_theta)**2)

        # store the lower bound and actual poisoned points
        real_target_lower_bound_and_attacks = []

        # default setting for target model is the actual model
        target_model.coef_= np.array([best_target_theta])
        target_model.intercept_ = np.array([best_target_bias])

        ##### Start the evaluation of different target classifiers #########
        if args.target_model == "real":
            print("------- Use Actual Target model as Target Model -----")
            if args.poison_whole:
                filename = 'files/online_models/{}/check_valid_thetas/whole-{}_online_data_theta-{}_tol-{}.npz'.format(dataset_name,cl_ind,\
                    i,args.incre_tol_par)
            else:
                filename = 'files/online_models/{}/subpop-{}_online_data_theta-{}_tol-{}.npz'.format(dataset_name,cl_ind,i,args.incre_tol_par)
            if not os.path.isfile(filename):
                # start the evaluation process
                online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
                best_max_loss_y, ol_tol_par, target_poison_max_losses, total_loss_diffs,\
                ol_tol_params, max_loss_diffs_reg, lower_bounds = incre_online_learning(X_train,
                                                                                        y_train,
                                                                                        curr_model,
                                                                                        target_model,
                                                                                        x_lim_tuples,
                                                                                        args,
                                                                                        ScikitModel,
                                                                                        target_model_type = "real",
                                                                                        attack_num_poison = 0,
                                                                                        kkt_tol_par = None)
                # retrain the online model based on poisons from our adaptive attack
                if len(online_poisons_y) > 0:
                    online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                    online_poisons_y = np.array(online_poisons_y)
                    online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                    online_full_y = np.concatenate((y_train,online_poisons_y),axis = 0)
                else:
                    print("online learning does not make progress and using original model!")
                    online_poisons_x = np.array(online_poisons_x)
                    online_poisons_y = np.array(online_poisons_y)
                    online_full_x = X_train
                    online_full_y = y_train
                print("shape of online poisoned points:",online_poisons_x.shape,online_poisons_y.shape)
                print("shape of full poisoned points:",online_full_x.shape,online_full_y.shape)
                # retrain the model based poisons from online learning
                C = 1.0 / (online_full_x.shape[0] * args.weight_decay)
                fit_intercept = True
                model_p_online = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 1000)
                model_p_online.fit(online_full_x, online_full_y) 

                # save the data and model for producing the online models
                if args.poison_whole:
                    filename = 'files/online_models/{}/check_valid_thetas/whole-{}_online_model_theta-{}_tol-{}.sav'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/check_valid_thetas/subpop-{}_online_model_theta-{}_tol-{}.sav'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                joblib.dump(model_p_online, filename)

                if args.poison_whole:
                    filename = 'files/online_models/{}/check_valid_thetas/whole-{}_online_data_theta-{}_tol-{}.npz'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/check_valid_thetas/subpop-{}_online_data_theta-{}_tol-{}.npz'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                np.savez(filename,
                        online_poisons_x = online_poisons_x,
                        online_poisons_y = online_poisons_y,
                        best_lower_bound = best_lower_bound,
                        conser_lower_bound = conser_lower_bound,
                        best_max_loss_x = best_max_loss_x,
                        best_max_loss_y = best_max_loss_y,
                        target_poison_max_losses = target_poison_max_losses,
                        total_loss_diffs = total_loss_diffs, 
                        max_loss_diffs = max_loss_diffs_reg,
                        lower_bounds = lower_bounds,
                        ol_tol_params = ol_tol_params
                        )
            else:
                if args.poison_whole:
                    filename = 'files/online_models/{}/whole-{}_online_data_theta-{}_tol-{}.npz'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/subpop-{}_online_data_theta-{}_tol-{}.npz'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                data_info = np.load(filename)
                online_poisons_x = data_info["online_poisons_x"]
                online_poisons_y = data_info["online_poisons_y"]
                best_lower_bound = data_info["best_lower_bound"]
                conser_lower_bound = data_info["conser_lower_bound"]
                best_max_loss_x = data_info["best_max_loss_x"]
                best_max_loss_y = data_info["best_max_loss_y"]
                target_poison_max_losses = data_info["target_poison_max_losses"]
                total_loss_diffs = data_info["total_loss_diffs"]
                max_loss_diffs_reg = data_info["max_loss_diffs"]
                lower_bounds = data_info["lower_bounds"]
                ol_tol_params = data_info["ol_tol_params"]  
                ol_tol_par = ol_tol_params[-1]

                if args.poison_whole:
                    filename = 'files/online_models/{}/check_valid_thetas/whole-{}_online_model_theta-{}_tol-{}.sav'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                else:
                    filename = 'files/online_models/{}/check_valid_thetas/subpop-{}_online_model_theta-{}_tol-{}.sav'.format(dataset_name,cl_ind,i,args.incre_tol_par)
                model_p_online = joblib.load(open(filename, 'rb'))

            # compute the norm difference
            target_model_b = target_model.intercept_
            target_model_b = target_model_b[0]
            model_p_online_b = model_p_online.intercept_
            model_p_online_b = model_p_online_b[0]
            norm_diff = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-model_p_online.coef_.reshape(-1))**2+(target_model_b - model_p_online_b)**2)
            # get prediction scores of produced model from adaptive online attack
            total_tst_acc = model_p_online.score(X_test, y_test)
            target_tst_acc = model_p_online.score(tst_sub_x, tst_sub_y)
            collat_tst_acc = model_p_online.score(tst_nsub_x,tst_nsub_y)
            total_trn_acc = model_p_online.score(X_train, y_train)
            target_trn_acc = model_p_online.score(trn_sub_x, trn_sub_y)
            collat_trn_acc = model_p_online.score(trn_nsub_x,trn_nsub_y)
            ol_acc_scores = [total_tst_acc,target_tst_acc,collat_tst_acc,total_trn_acc,target_trn_acc,collat_trn_acc]

            # key attack statistics are stored here
            real_target_lower_bound_and_attacks = real_target_lower_bound_and_attacks + [loss_Dc,best_lower_bound,conser_lower_bound,len(online_poisons_y),
            ol_tol_par,norm_diff] + orig_model_acc_scores + target_model_acc_scores + ol_acc_scores
            # write key attack info to the csv files
            check_valid_thetas_writer.writerow(real_target_lower_bound_and_attacks)

    # close all files
    check_valid_thetas_file.flush()
    check_valid_thetas_file.close()
