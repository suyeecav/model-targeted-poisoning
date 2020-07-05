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
parser.add_argument('--target_model', default='all',help='set your target classifier, options: kkt, ol, real, compare, all')
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, 2d_toy")
parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')

# some params related to online algorithm, use the default
parser.add_argument('--online_alg_criteria',default='max_loss',help='stop criteria of online alg: max_loss or norm')
parser.add_argument('--incre_tol_par',default=1e-2,type=float,help='stop value of online alg: max_loss or norm')
parser.add_argument('--weight_decay',default=0.09,type=float,help='weight decay for regularizers')
parser.add_argument('--err_threshold',default=1.0,type=float,help='target error rate')
parser.add_argument('--rand_seed',default=12,type=int,help='random seed')
parser.add_argument('--repeat_num',default=1,type=int,help='repeat num of maximum loss diff point')
parser.add_argument('--improved',action="store_true",help='if true, target classifier is obtained through improved process')
parser.add_argument('--fixed_budget',default=0,type=int,help='if > 0, then run the attack for fixed number of points')

args = parser.parse_args()

####################### set up the poisoning attack parameters #######################################

# KKT attack specific parameters
percentile = 90
loss_percentile = 90
use_slab = False
use_loss = False
use_l2 = False
dataset_name = args.dataset
# if true, we generate target classifier using label flipping...
if args.improved:
    target_gen_proc = 'improved'
else:
    target_gen_proc = 'orig'

assert dataset_name in ['adult','mnist_17','2d_toy']
if dataset_name == 'mnist_17':
    args.poison_whole = True
    # see if decreasing by half helps
    args.incre_tol_par = 0.1
    # args.weight_decay = 0.09
    valid_theta_errs = [0.05,0.1,0.15]
elif dataset_name == 'adult':
    # args.incre_tol_par = 0.01
    valid_theta_errs = [1.0]
elif dataset_name == '2d_toy':
    args.poison_whole = True
    if args.poison_whole:
        valid_theta_errs = [0.1,0.15] 
    else:
        valid_theta_errs = [1.0]

if args.model_type == 'svm':
    print("chosen model: svm")
    ScikitModel = svm_model
    model_grad = hinge_grad
else:
    print("chosen model: lr")
    ScikitModel = logistic_model
    model_grad = logistic_grad

learning_rate = 0.01
######################################################################

################# Main body of work ###################
# creat files that store clustering info
make_dirs(args)

# load data
X_train, y_train, X_test, y_test = load_dataset(args.dataset)

if min(y_test)>-1:
    y_test = 2*y_test-1
if min(y_train) > -1:
    y_train = 2*y_train - 1

full_x = np.concatenate((X_train,X_test),axis=0)
full_y = np.concatenate((y_train,y_test),axis=0)
if args.dataset == "2d_toy":
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
            random_state=args.rand_seed,
            verbose=False,
            max_iter = 1000)
model.fit(X_train, y_train)

# some models defined only to use as an instance of classification model
model_dumb = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter = 1000)
model_dumb.fit(X_train, y_train)

model_dumb1 = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
            verbose=False,
            max_iter = 1000)
model_dumb1.fit(X_train[0:2000], y_train[0:2000])

# will be used as the model generated from the KKT attack
kkt_model_p = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            random_state=args.rand_seed,
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

# start the complete process 
for valid_theta_err in valid_theta_errs:
    print("Attack Target Classifiers with Expected Error Rate:",valid_theta_err)
    args.err_threshold = valid_theta_err
    # open the files to write key info
    if args.target_model == "all":
        kkt_lower_bound_file = open('files/results/{}/{}/{}/{}/kkt_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        kkt_lower_bound_writer = csv.writer(kkt_lower_bound_file, delimiter=str(' ')) 

        real_lower_bound_file = open('files/results/{}/{}/{}/{}/real_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        real_lower_bound_writer = csv.writer(real_lower_bound_file, delimiter=str(' ')) 

        ol_lower_bound_file = open('files/results/{}/{}/{}/{}/ol_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        ol_lower_bound_writer = csv.writer(ol_lower_bound_file, delimiter=str(' ')) 
    elif args.target_model == "kkt":
        kkt_lower_bound_file = open('files/results/{}/{}/{}/{}/kkt_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        kkt_lower_bound_writer = csv.writer(kkt_lower_bound_file, delimiter=str(' ')) 
    elif args.target_model == "real":
        real_lower_bound_file = open('files/results/{}/{}/{}/{}/real_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        real_lower_bound_writer = csv.writer(real_lower_bound_file, delimiter=str(' ')) 
    elif args.target_model == "ol":
        ol_lower_bound_file = open('files/results/{}/{}/{}/{}/ol_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        ol_lower_bound_writer = csv.writer(ol_lower_bound_file, delimiter=str(' ')) 
    elif args.target_model == "compare":
        compare_lower_bound_file = open('files/results/{}/{}/{}/{}/compare_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,args.incre_tol_par,valid_theta_err), 'w')
        compare_lower_bound_writer = csv.writer(compare_lower_bound_file, delimiter=str(' ')) 

    for kk in range(len(cl_inds)):
        cl_ind = int(cl_inds[kk])
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
        if not args.improved:
            if args.poison_whole:
                fname = open('files/target_classifiers/{}/orig_best_theta_whole_err-{}'.format(dataset_name,valid_theta_err), 'rb')  
                
            else:
                fname = open('files/target_classifiers/{}/orig_best_theta_subpop_{}_err-{}'.format(dataset_name,cl_ind,valid_theta_err), 'rb')  
        else:
            if args.poison_whole:
                fname = open('files/target_classifiers/{}/improved_best_theta_whole_err-{}'.format(dataset_name,valid_theta_err), 'rb')  
                
            else:
                fname = open('files/target_classifiers/{}/improved_best_theta_subpop_{}_err-{}'.format(dataset_name,cl_ind,valid_theta_err), 'rb')  
        f = pickle.load(fname)
        best_target_theta = f['thetas']
        best_target_bias = f['biases']
        sub_frac = 1

        # fname = 'files/target_classifiers/{}/orig_best_poison_subpop_{}_err-{}.npz'.format(dataset_name,int(cl_ind),valid_theta_err)
        # poisons_all = np.load(fname)
        # X_Poison = poisons_all["X_poison"]
        # Y_Poison = poisons_all["Y_poison"]
        # no longer need it, so just fill some random stuff
        poisons_all = {}
        poisons_all["X_poison"] = X_train
        poisons_all["Y_poison"] = y_train

        # # print info of the target classifier # #
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
        _, ideal_total_err = calculate_loss(margins)
        print("Ideal Total Train Acc:",1-ideal_total_err)
        print("Ideal Target Train Acc:",1-ideal_target_err)
        print("Ideal Collat Train Acc:",1-ideal_collat_err)
        target_model_acc_scores.append(1-ideal_total_err)
        target_model_acc_scores.append(1-ideal_target_err)
        target_model_acc_scores.append(1-ideal_collat_err)

        # # just to make sure one subpop of 2d toy example will terminate
        # if 1-ideal_target_err > 1-args.err_threshold: 
        #     print("the target classifier does not satisfy the attack goal, skip the rest!")
        #     continue
        # store the lower bound and actual poisoned points
        kkt_target_lower_bound_and_attacks = []
        ol_target_lower_bound_and_attacks = []
        real_target_lower_bound_and_attacks = []
        compare_target_lower_bound_and_attacks = []

        print("************** Target Classifier for Subpop:{} ***************".format(cl_ind))
        if not fit_intercept:
            target_bias = 0

        ## apply online learning algorithm to provide lower bound and candidate attack ##
        C = 1.0 / (X_train.shape[0] * args.weight_decay)
        curr_model = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=args.rand_seed,
                    verbose=False,
                    max_iter = 1000)
        curr_model.fit(X_train, y_train)

        target_model = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=args.rand_seed,
                    verbose=False,
                    max_iter = 1000)
        target_model.fit(X_train, y_train)
        # default setting for target model is the actual model
        target_model.coef_= np.array([best_target_theta])
        target_model.intercept_ = np.array([best_target_bias])
        ##### Start the evaluation of different target classifiers #########
        if args.target_model == "real" or args.target_model == "all":
            print("------- Use Actual Target model as Target Model -----")
            if args.poison_whole:
                filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            else:
                filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            if not os.path.isfile(filename):
                # start the evaluation process
                print("[Sanity Real] Acc of current model:",curr_model.score(X_test,y_test),curr_model.score(X_train,y_train))

                online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
                best_max_loss_y, ol_tol_par, target_poison_max_losses, current_total_losses,\
                ol_tol_params, max_loss_diffs_reg, lower_bounds, online_acc_scores,norm_diffs = incre_online_learning(X_train,
                                                                                        y_train,
                                                                                        X_test,
                                                                                        y_test,
                                                                                        curr_model,
                                                                                        target_model,
                                                                                        x_lim_tuples,
                                                                                        args,
                                                                                        ScikitModel,
                                                                                        target_model_type = "real",
                                                                                        attack_num_poison = 0,
                                                                                        kkt_tol_par = None,
                                                                                        subpop_data = subpop_data,
                                                                                        target_poisons = poisons_all)
                # retrain the online model based on poisons from our adaptive attack
                if len(online_poisons_y) > 0:
                    print("Original Shape of online x and online y:",np.array(online_poisons_x).shape,np.array(online_poisons_y).shape)
                    online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                    online_poisons_y = np.concatenate(online_poisons_y,axis=0)
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
                    random_state=args.rand_seed,
                    verbose=False,
                    max_iter = 1000)
                model_p_online.fit(online_full_x, online_full_y) 

                # save the data and model for producing the online models
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                joblib.dump(model_p_online, filename)

                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                np.savez(filename,
                        online_poisons_x = online_poisons_x,
                        online_poisons_y = online_poisons_y,
                        best_lower_bound = best_lower_bound,
                        conser_lower_bound = conser_lower_bound,
                        best_max_loss_x = best_max_loss_x,
                        best_max_loss_y = best_max_loss_y,
                        target_poison_max_losses = target_poison_max_losses,
                        current_total_losses = current_total_losses, 
                        max_loss_diffs = max_loss_diffs_reg,
                        lower_bounds = lower_bounds,
                        ol_tol_params = ol_tol_params,
                        online_acc_scores = np.array(online_acc_scores),
                        norm_diffs = norm_diffs
                        )
            else:
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                data_info = np.load(filename)
                online_poisons_x = data_info["online_poisons_x"]
                online_poisons_y = data_info["online_poisons_y"]
                best_lower_bound = data_info["best_lower_bound"]
                conser_lower_bound = data_info["conser_lower_bound"]
                best_max_loss_x = data_info["best_max_loss_x"]
                best_max_loss_y = data_info["best_max_loss_y"]
                target_poison_max_losses = data_info["target_poison_max_losses"]
                current_total_losses = data_info["current_total_losses"]
                max_loss_diffs_reg = data_info["max_loss_diffs"]
                lower_bounds = data_info["lower_bounds"]
                ol_tol_params = data_info["ol_tol_params"]  
                ol_tol_par = ol_tol_params[-1]
                online_acc_scores = data_info["online_acc_scores"]
                norm_diffs = data_info['norm_diffs']
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                model_p_online = joblib.load(open(filename, 'rb'))

            ###  perform the KKT attack with same number of poisned points of our Adaptive attack ###
            kkt_fraction = 1
            if args.poison_whole:
                filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
            else:
                filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
            if not os.path.isfile(filename):
                target_theta = np.copy(best_target_theta)
                if fit_intercept:
                    target_bias = np.copy(best_target_bias)
                else:
                    target_bias = 0
                # sanity check for target model info
                print("--- Sanity Check Info for Subpop ---")
                margins = tst_sub_y*(tst_sub_x.dot(target_theta) + target_bias)
                _, ideal_target_err = calculate_loss(margins)
                margins =tst_nsub_y*(tst_nsub_x.dot(target_theta) + target_bias)
                _, ideal_collat_err = calculate_loss(margins)
                margins =y_test*(X_test.dot(target_theta) + target_bias)
                _, ideal_total_err = calculate_loss(margins)
                print("Ideal Total Test Acc:",1-ideal_total_err)
                print("Ideal Target Test Acc:",1-ideal_target_err)
                print("Ideal Collat Test Acc:",1-ideal_collat_err)
                margins = trn_sub_y*(trn_sub_x.dot(target_theta) + target_bias)
                _, ideal_target_err = calculate_loss(margins)
                margins =trn_nsub_y*(trn_nsub_x.dot(target_theta) + target_bias)
                _, ideal_collat_err = calculate_loss(margins)
                margins =y_train*(X_train.dot(target_theta) + target_bias)
                _, ideal_total_err = calculate_loss(margins)
                print("Ideal Total Train Acc:",1-ideal_total_err)
                print("Ideal Target Train Acc:",1-ideal_target_err)
                print("Ideal Collat Train Acc:",1-ideal_collat_err)

                # also explore some of the KKT attack results using smaller num of poisons 
                kkt_fractions = [0.2,0.4,0.6,0.8,1]
                kkt_fraction_max_loss_diffs = []
                kkt_fraction_norm_diffs = []
                kkt_fraction_acc_scores = []
                kkt_fraction_num_poisons = []
                kkt_fraction_loss_on_clean = []
                # setup the kkt attack class
                two_class_kkt, clean_grad_at_target_theta, target_bias_grad, max_losses = kkt_attack.kkt_setup(
                    target_theta,
                    target_bias,
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

                for kkt_fraction in kkt_fractions:
                    # eps pairs and objective for choosing best kkt classifier
                    epsilon_pairs = []
                    best_grad_diff_norm = 1e10

                    kkt_num_points = int(len(online_poisons_y)*kkt_fraction)
                    kkt_fraction_num_poisons.append(kkt_num_points)
                    total_epsilon = float(kkt_num_points)/X_train.shape[0]
                    print("Explored kkt fraction {}, number of poisons for kkt attack {}, the poison ratio {}".format(kkt_fraction,kkt_num_points,total_epsilon))
                    model_dumb1.coef_ = np.array([target_theta])
                    model_dumb1.intercept_ = np.array([target_bias]) 

                    target_grad = clean_grad_at_target_theta + ((1 + total_epsilon) * args.weight_decay * target_theta)
                    epsilon_neg = (total_epsilon - target_bias_grad) / 2
                    epsilon_pos = total_epsilon - epsilon_neg

                    if (epsilon_neg >= 0) and (epsilon_neg <= total_epsilon):
                        epsilon_pairs.append((epsilon_pos, epsilon_neg))

                    for epsilon_pos in np.arange(0, total_epsilon + 1e-6, epsilon_increment):
                        epsilon_neg = total_epsilon - epsilon_pos
                        epsilon_pairs.append((epsilon_pos, epsilon_neg))
                    
                    for epsilon_pos, epsilon_neg in epsilon_pairs:
                        print('\n## Trying epsilon_pos %s, epsilon_neg %s' % (epsilon_pos, epsilon_neg))
                        X_modified, Y_modified, obj, x_pos, x, num_pos, num_neg = kkt_attack.kkt_attack(
                            two_class_kkt,
                            target_grad, target_theta,
                            total_epsilon * sub_frac, epsilon_pos * sub_frac, epsilon_neg * sub_frac,
                            X_train_cp, y_train_cp,
                            class_map, centroids, centroid_vec, sphere_radii, slab_radii,
                            target_bias, target_bias_grad, max_losses)
                        
                        # separate out the poisoned points
                        idx_poison = slice(X_train.shape[0], X_modified.shape[0])
                        idx_clean = slice(0, X_train.shape[0])
                        
                        X_poison = X_modified[idx_poison,:]
                        Y_poison = Y_modified[idx_poison]   
                        # unique points and labels in kkt attack
                        unique_x, unique_indices, unique_counts = np.unique(X_poison,return_index = True,return_counts = True,axis=0)
                        unique_y = Y_poison[unique_indices]               
                        # retrain the model 
                        C = 1.0 / (X_modified.shape[0] * args.weight_decay)
                        model_p = ScikitModel(
                            C=C,
                            tol=1e-8,
                            fit_intercept=fit_intercept,
                            random_state=args.rand_seed,
                            verbose=False,
                            max_iter = 1000)
                        model_p.fit(X_modified, Y_modified)                 
                        # acc on subpop and rest of pops
                        trn_total_acc = model_p.score(X_train, y_train)
                        trn_target_acc = model_p.score(trn_sub_x, trn_sub_y)
                        trn_collat_acc = model_p.score(trn_nsub_x, trn_nsub_y)
                        tst_total_acc = model_p.score(X_test, y_test)
                        tst_target_acc = model_p.score(tst_sub_x, tst_sub_y)
                        tst_collat_acc = model_p.score(tst_nsub_x, tst_nsub_y)
                        print()
                        
                        print('Test Total Acc : ', tst_total_acc)
                        print('Test Target Acc : ', tst_target_acc)
                        print('Test Collat Acc : ', tst_collat_acc)
                        print('Train Total Acc : ', trn_total_acc)
                        print('Train Target Acc : ', trn_target_acc)
                        print('Train Collat Acc : ', trn_collat_acc)

                        # sanity check on the max loss difference between target model and kkt model
                        kkt_tol_par = -1
                        for y_b in set(y_train):
                            if y_b == 1:
                                max_loss_diff,_ = search_max_loss_pt(model_p,model_dumb1,y_b,x_pos_tuple,args)
                                if kkt_tol_par < max_loss_diff:
                                    kkt_tol_par = max_loss_diff
                            elif y_b == -1:
                                max_loss_diff,_ = search_max_loss_pt(model_p,model_dumb1,y_b,x_neg_tuple,args)
                                if kkt_tol_par < max_loss_diff:
                                    kkt_tol_par = max_loss_diff
                        print("max loss difference between target and kkt model is:",kkt_tol_par)
                        model_dumb1_b = model_dumb1.intercept_
                        model_dumb1_b = model_dumb1_b[0]
                        model_p_b = model_p.intercept_
                        model_p_b = model_p_b[0]
                        kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-model_p.coef_.reshape(-1))**2+(model_dumb1_b - model_p_b)**2)
                        print("norm difference between target and kkt model is:",kkt_tol_par_norm)
                        if obj < best_grad_diff_norm:
                            best_grad_diff_norm = obj
                            # used for theoretical lower bound computation
                            kkt_model_p.coef_ = np.copy(model_p.coef_)
                            kkt_model_p.intercept_ = np.copy(model_p.intercept_)
                            # best_target_acc1 = tst_target_acc
                            kkt_unique_x = np.copy(unique_x)
                            kkt_unique_y = np.copy(unique_y)
                            kkt_unique_counts = np.copy(unique_counts)
                            kkt_x_modified = np.copy(X_modified)
                            kkt_y_modified = np.copy(Y_modified)
                            # store the best statistics
                            best_max_loss_diff = np.copy(kkt_tol_par)
                            best_norm_diff = np.copy(kkt_tol_par_norm)
                            best_acc_scores = [tst_total_acc,tst_target_acc,tst_collat_acc,trn_total_acc,trn_target_acc,trn_collat_acc]

                    # save all the kkt fraction related info
                    kkt_fraction_max_loss_diffs.append(best_max_loss_diff)
                    kkt_fraction_norm_diffs.append(best_norm_diff)
                    kkt_fraction_acc_scores.append(best_acc_scores)
                    # compute the loss of kkt model on clean train set
                    aaaa = kkt_model_p.intercept_
                    c_margins = y_train*(X_train.dot(kkt_model_p.coef_.reshape(-1)) + aaaa[0])
                    kkt_loss_on_clean = np.sum(np.maximum(1-c_margins, 0))
                    kkt_fraction_loss_on_clean.append(kkt_loss_on_clean)

                    # save the model weights and the train and test data
                    if args.poison_whole:
                        filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                    else:
                        filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                    joblib.dump(kkt_model_p, filename)
                    if args.poison_whole:
                        filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                    else:
                        filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                    best_kkt_theta = kkt_model_p.coef_.reshape(-1)
                    best_kkt_bias = kkt_model_p.intercept_
                    best_kkt_bias = best_kkt_bias[0]
                    np.savez(filename,
                            kkt_x_modified = kkt_x_modified,
                            kkt_y_modified = kkt_y_modified,
                            kkt_unique_x = kkt_unique_x,
                            kkt_unique_y = kkt_unique_y,
                            kkt_unique_counts = kkt_unique_counts,
                            best_target_theta = best_target_theta,
                            best_target_bias = best_target_bias,
                            best_kkt_theta = best_kkt_theta,
                            best_kkt_bias = best_kkt_bias
                            )
                # store the kkt fraction info
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_kkt_frac_info_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_kkt_frac_info_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                np.savez(filename,
                        kkt_fraction_max_loss_diffs = np.array(kkt_fraction_max_loss_diffs),
                        kkt_fraction_norm_diffs = np.array(kkt_fraction_norm_diffs),
                        kkt_fraction_acc_scores = np.array(kkt_fraction_acc_scores),
                        kkt_fraction_num_poisons = np.array(kkt_fraction_num_poisons),
                        kkt_fraction_loss_on_clean = np.array(kkt_fraction_loss_on_clean)
                        )

                print("--- Some info of KKT attack with smaller num of poisons ---")
                a = np.array(kkt_fraction_num_poisons)
                print(kkt_fraction_num_poisons)
                print(kkt_fraction_max_loss_diffs)
                print(kkt_fraction_norm_diffs)
                print(kkt_fraction_acc_scores)
                print("--- Corresponding info of KKT attack with smaller num of poisons ---")
                print(np.array(ol_tol_params)[a/args.repeat_num])
                print(np.array(norm_diffs)[a/args.repeat_num])
                print("the online attack acc scores")
                tmp_trn_acc,tmp_trn_sub_acc,tmp_trn_nsub_acc,tmp_tst_acc,tmp_tst_sub_acc,tmp_tst_nsub_acc = \
                    online_acc_scores[0],online_acc_scores[1],online_acc_scores[2],online_acc_scores[3],\
                        online_acc_scores[4],online_acc_scores[5]
                print(np.array(tmp_tst_acc)[a/args.repeat_num])
                print(np.array(tmp_tst_sub_acc)[a/args.repeat_num])
                print(np.array(tmp_tst_nsub_acc)[a/args.repeat_num])
                print(np.array(tmp_trn_acc)[a/args.repeat_num])
                print(np.array(tmp_trn_sub_acc)[a/args.repeat_num])
                print(np.array(tmp_trn_nsub_acc)[a/args.repeat_num])
                
            else:
                # when loading the files, only load the full kkt attack
                # load the kkt related model and data 
                kkt_fraction = 1
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                else:
                    filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                kkt_model_p = joblib.load(open(filename, 'rb'))
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                else:
                    filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                data_info = np.load(filename)
                kkt_x_modified = data_info["kkt_x_modified"]
                kkt_y_modified = data_info["kkt_y_modified"]
                kkt_unique_x = data_info["kkt_unique_x"]
                kkt_unique_y = data_info["kkt_unique_y"]
                kkt_unique_counts = data_info["kkt_unique_counts"]
                # best_target_theta = data_info["best_target_theta"]
                # best_target_bias = data_info["best_target_bias"]
                idx_poison = slice(X_train.shape[0], kkt_x_modified.shape[0])
                idx_clean = slice(0, X_train.shape[0])
                total_epsilon = float(len(online_poisons_y))/X_train.shape[0]

                tst_target_acc = kkt_model_p.score(tst_sub_x, tst_sub_y)
                tst_collat_acc = kkt_model_p.score(tst_nsub_x, tst_nsub_y)
                print("--------Performance of Selected KKT attack model-------")
                # print('Total Train Acc : %.3f' % kkt_model_p.score(X_train, y_train))
                print('Total Test Acc : ', kkt_model_p.score(X_test, y_test))
                print('Test Target Acc : ', tst_target_acc)
                print('Test Collat Acc : ', tst_collat_acc)
                print('Total Train Acc : ', kkt_model_p.score(X_train, y_train))
                print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
                print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))

            # sanity check for kkt points 
            assert np.array_equal(X_train, kkt_x_modified[idx_clean,:])
            assert np.array_equal(y_train,kkt_y_modified[idx_clean])

            print("min and max feature values of kkt points:")
            poison_xs = kkt_x_modified[idx_poison]
            poison_ys = kkt_y_modified[idx_poison]
            print(np.amax(poison_xs),np.amin(poison_xs))
            print(np.amax(poison_ys),np.amin(poison_ys))
            x_pos_min, x_pos_max = x_pos_tuple
            print(np.amax(poison_xs),x_pos_max)
            if dataset_name != '2d_toy':
                assert np.amax(poison_xs) <= (x_pos_max + float(x_pos_max)/100)
                assert np.amin(poison_xs) >= (x_pos_min - np.abs(float(x_pos_min))/100)

            # sanity check on the max loss difference between target model and kkt model
            print("----- Info of the Selected kkt model ---")
            print('Test Total Acc : ', kkt_model_p.score(X_test, y_test))
            print('Test Target Acc : ', kkt_model_p.score(tst_sub_x,tst_sub_y))
            print('Test Collat Acc : ', kkt_model_p.score(tst_nsub_x,tst_nsub_y))
            print('Train Total Acc : ', kkt_model_p.score(X_train, y_train))
            print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
            print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))

            model_dumb1.coef_ = np.array([best_target_theta])
            model_dumb1.intercept_ = np.array([best_target_bias])
            kkt_tol_par = -1
            for y_b in set(y_train):
                if y_b == 1:
                    max_loss_diff,_ = search_max_loss_pt(kkt_model_p,model_dumb1,y_b,x_pos_tuple,args)
                    if kkt_tol_par < max_loss_diff:
                        kkt_tol_par = max_loss_diff
                elif y_b == -1:
                    max_loss_diff,_ = search_max_loss_pt(kkt_model_p,model_dumb1,y_b,x_neg_tuple,args)
                    if kkt_tol_par < max_loss_diff:
                        kkt_tol_par = max_loss_diff
            print("max loss difference between selected target and selected kkt model is:",kkt_tol_par)
            # assert np.array_equal(model_dumb1.coef_,target_model.coef_)
            # assert np.array_equal(model_dumb1.intercept_,target_model.intercept_)
            model_dumb1_b = model_dumb1.intercept_
            model_dumb1_b = model_dumb1_b[0]
            kkt_model_p_b = kkt_model_p.intercept_
            kkt_model_p_b = kkt_model_p_b[0]
            kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-kkt_model_p.coef_.reshape(-1))**2+(model_dumb1_b - kkt_model_p_b)**2)
            print("norm difference between selected target and selected kkt model is:",kkt_tol_par_norm)
            if kkt_tol_par < 1e-4:
                print("something wrong with selected kkt model or the target model!")
                sys.exit(0)

            # compute the grad norm difference and store the value
            kkt_grad_norm_diff = compute_grad_norm_diff(best_target_theta,best_target_bias,total_epsilon,\
                X_train,y_train,poison_xs,poison_ys,args)
            ol_grad_norm_diff = compute_grad_norm_diff(best_target_theta,best_target_bias,total_epsilon,\
                X_train,y_train,online_poisons_x,online_poisons_y,args)
            print("Grad norm difference of KKT to target:",kkt_grad_norm_diff)
            print("Grad norm difference of adaptive to target:",ol_grad_norm_diff)
            
            # Print the lower bound and performance of different attacks 
            final_norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                                X_train,
                                                                y_train,
                                                                X_test,
                                                                y_test,
                                                                subpop_data,
                                                                best_lower_bound,
                                                                conser_lower_bound,
                                                                kkt_tol_par,
                                                                ol_tol_par,
                                                                target_model,
                                                                kkt_model_p,
                                                                model_p_online,
                                                                len(online_poisons_y),
                                                                args)
            # key attack statistics are stored here
            real_target_lower_bound_and_attacks = real_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,len(online_poisons_y),len(online_poisons_y),
            kkt_tol_par, ol_tol_par] + final_norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores + [kkt_grad_norm_diff,ol_grad_norm_diff]
            # write key attack info to the csv files
            real_lower_bound_writer.writerow(real_target_lower_bound_and_attacks)
            
        if  args.target_model == "kkt" or args.target_model == "all":
            print("------- Use KKT model as Target Model -----")
            # load the kkt related model and data
            kkt_fraction = 1  
            if args.poison_whole:
                filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
            else:
                filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
            kkt_model_p = joblib.load(open(filename, 'rb'))
            if args.poison_whole:
                filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
            else:
                filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
            data_info = np.load(filename)
            kkt_x_modified = data_info["kkt_x_modified"]
            kkt_y_modified = data_info["kkt_y_modified"]
            kkt_unique_x = data_info["kkt_unique_x"]
            kkt_unique_y = data_info["kkt_unique_y"]
            kkt_unique_counts = data_info["kkt_unique_counts"]
            # best_target_theta = data_info["best_target_theta"]
            # best_target_bias = data_info["best_target_bias"]
            idx_poison = slice(X_train.shape[0], kkt_x_modified.shape[0])
            idx_clean = slice(0, X_train.shape[0])

            tst_target_acc = kkt_model_p.score(tst_sub_x, tst_sub_y)
            tst_collat_acc = kkt_model_p.score(tst_nsub_x, tst_nsub_y)
            print("--------Performance of Selected KKT attack model-------")
            # print('Total Train Acc : %.3f' % kkt_model_p.score(X_train, y_train))
            print('Total Test Acc : ', kkt_model_p.score(X_test, y_test))
            print('Test Target Acc : ', tst_target_acc)
            print('Test Collat Acc : ', tst_collat_acc)
            print('Total Train Acc : ', kkt_model_p.score(X_train, y_train))
            print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
            print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))

            target_model.coef_ = kkt_model_p.coef_
            target_model.intercept_ = kkt_model_p.intercept_
            kkt_tol_par = 0
            # check if adaptive online attack is needed
            if args.poison_whole:
                filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            else:
                filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            if not os.path.isfile(filename):
                # start the lower bound computation process
                # reset current model to clean model, just to measure the impact of curr_model
                # curr_model.fit(X_train, y_train)
                
                print("[Sanity KKT] Acc of current model:",curr_model.score(X_test,y_test),curr_model.score(X_train,y_train))

                online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
                best_max_loss_y, ol_tol_par, target_poison_max_losses, current_total_losses,\
                ol_tol_params, max_loss_diffs_reg, lower_bounds, online_acc_scores,norm_diffs = incre_online_learning(X_train,
                                                                                        y_train,
                                                                                        X_test,
                                                                                        y_test,
                                                                                        curr_model,
                                                                                        target_model,
                                                                                        x_lim_tuples,
                                                                                        args,
                                                                                        ScikitModel,
                                                                                        target_model_type = "kkt",
                                                                                        attack_num_poison = kkt_x_modified.shape[0]-X_train.shape[0],
                                                                                        kkt_tol_par = kkt_tol_par,
                                                                                        subpop_data = subpop_data,
                                                                                        target_poisons = poisons_all)
                # retrain the online model based on poisons from our adaptive attack
                if len(online_poisons_y) > 0:
                    online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                    online_poisons_y = np.concatenate(online_poisons_y,axis=0)
                    online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                    print(y_train.shape,online_poisons_y.shape)
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
                    random_state=args.rand_seed,
                    verbose=False,
                    max_iter = 1000)
                model_p_online.fit(online_full_x, online_full_y) 

                # need to save the posioned model from our attack, for the purpose of validating the lower bound for the online attack
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_kkt_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_kkt_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                joblib.dump(model_p_online, filename)
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                np.savez(filename,
                        online_poisons_x = online_poisons_x,
                        online_poisons_y = online_poisons_y,
                        best_lower_bound = best_lower_bound,
                        conser_lower_bound = conser_lower_bound,
                        best_max_loss_x = best_max_loss_x,
                        best_max_loss_y = best_max_loss_y,
                        target_poison_max_losses = target_poison_max_losses,
                        current_total_losses = current_total_losses, 
                        max_loss_diffs = max_loss_diffs_reg,
                        lower_bounds = lower_bounds,
                        ol_tol_params = ol_tol_params,
                        online_acc_scores = np.array(online_acc_scores),
                        norm_diffs = norm_diffs
                        )
            else:
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                data_info = np.load(filename)
                online_poisons_x = data_info["online_poisons_x"]
                online_poisons_y = data_info["online_poisons_y"]
                best_lower_bound = data_info["best_lower_bound"]
                conser_lower_bound = data_info["conser_lower_bound"]
                best_max_loss_x = data_info["best_max_loss_x"]
                best_max_loss_y = data_info["best_max_loss_y"]
                target_poison_max_losses = data_info["target_poison_max_losses"]
                current_total_losses = data_info["current_total_losses"]
                max_loss_diffs_reg = data_info["max_loss_diffs"]
                lower_bounds = data_info["lower_bounds"]
                ol_tol_params = data_info["ol_tol_params"]  
                ol_tol_par = ol_tol_params[-1]
                online_acc_scores = data_info['online_acc_scores']
                norm_diffs = data_info['norm_diffs']

                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_kkt_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_kkt_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                model_p_online = joblib.load(open(filename, 'rb'))

            # summarize the attack results
            final_norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                                X_train,
                                                                y_train,
                                                                X_test,
                                                                y_test,
                                                                subpop_data,
                                                                best_lower_bound,
                                                                conser_lower_bound,
                                                                kkt_tol_par,
                                                                ol_tol_par,
                                                                target_model,
                                                                kkt_model_p,
                                                                model_p_online,
                                                                kkt_x_modified.shape[0]-X_train.shape[0],
                                                                args)
            if best_lower_bound > (kkt_x_modified.shape[0]-X_train.shape[0]):
                print("violation observed for the lower bound of KKT model!")
                sys.exit(0)
            kkt_target_lower_bound_and_attacks = kkt_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,\
                kkt_x_modified.shape[0]-X_train.shape[0],len(online_poisons_y),
            kkt_tol_par, ol_tol_par] + final_norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores
            # write key attack info to the csv files
            kkt_lower_bound_writer.writerow(kkt_target_lower_bound_and_attacks)

        if args.target_model == "ol" or args.target_model == "all":
            print("------- Use Aptive Poison model as Target Model -----")
            # load the adaptive attack models
            if args.poison_whole:
                filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            else:
                filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_real_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            target_model_p_online = joblib.load(open(filename, 'rb'))
            if args.poison_whole:
                filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            else:
                filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            data_info = np.load(filename)
            target_online_poisons_x = data_info["online_poisons_x"]
            target_online_poisons_y = data_info["online_poisons_y"]
            target_online_full_x = np.concatenate((X_train,target_online_poisons_x),axis = 0)
            target_online_full_y = np.concatenate((y_train,target_online_poisons_y),axis = 0)

            if args.poison_whole:
                filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            else:
                filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
            if not os.path.isfile(filename):
                # validate it only when adaptive attack did execute for a while
                if len(target_online_poisons_y) > 0:
                    # separate out the poisoned points
                    idx_poison = slice(X_train.shape[0], target_online_full_x.shape[0])
                    idx_clean = slice(0, X_train.shape[0])
                    assert np.array_equal(X_train, target_online_full_x[idx_clean,:])
                    assert np.array_equal(y_train, target_online_full_y[idx_clean])    
                    target_model.coef_ = target_model_p_online.coef_
                    target_model.intercept_ = target_model_p_online.intercept_
                    # load the kkt model and get the kkt stop criteria
                    kkt_fraction = 1
                    if args.poison_whole:
                        filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                    else:
                        filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                    kkt_model_p = joblib.load(open(filename, 'rb'))
                    kkt_tol_par_max_loss = -1
                    for y_b in set(y_train):
                        if y_b == 1:
                            max_loss_diff,_ = search_max_loss_pt(kkt_model_p,target_model,y_b,x_pos_tuple,args)
                            if kkt_tol_par_max_loss < max_loss_diff:
                                kkt_tol_par_max_loss = max_loss_diff
                        elif y_b == -1:
                            max_loss_diff,_ = search_max_loss_pt(kkt_model_p,target_model,y_b,x_neg_tuple,args)
                            if kkt_tol_par_max_loss < max_loss_diff:
                                kkt_tol_par_max_loss = max_loss_diff
                    model_dumb1_b = model_dumb1.intercept_
                    model_dumb1_b = model_dumb1_b[0]
                    kkt_model_p_b = kkt_model_p.intercept_
                    kkt_model_p_b = kkt_model_p_b[0]
                    kkt_tol_par_norm = np.sqrt(np.linalg.norm(model_dumb1.coef_.reshape(-1)-kkt_model_p.coef_.reshape(-1))**2+(model_dumb1_b - kkt_model_p_b)**2)
                    if args.online_alg_criteria == "max_loss":
                        kkt_tol_par = kkt_tol_par_max_loss
                    elif args.online_alg_criteria == "norm":
                        kkt_tol_par = kkt_tol_par_norm
                    print("max loss and norm criterias of kkt attack:",kkt_tol_par_max_loss,kkt_tol_par_norm)

                    # reset current model to clean model
                    # curr_model.fit(X_train, y_train)
                    print("[Sanity OL] Acc of current model:",curr_model.score(X_test,y_test),curr_model.score(X_train,y_train))

                    # start the evaluation process
                    online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
                    best_max_loss_y, ol_tol_par, target_poison_max_losses, current_total_losses,\
                    ol_tol_params, max_loss_diffs_reg, lower_bounds, online_acc_scores,norm_diffs = incre_online_learning(X_train,
                                                                                            y_train,
                                                                                            X_test,
                                                                                            y_test,
                                                                                            curr_model,
                                                                                            target_model,
                                                                                            x_lim_tuples,
                                                                                            args,
                                                                                            ScikitModel,
                                                                                            target_model_type = "ol",
                                                                                            attack_num_poison = len(target_online_poisons_y),
                                                                                            kkt_tol_par = kkt_tol_par,
                                                                                            subpop_data = subpop_data,
                                                                                            target_poisons = poisons_all)

                    # retrain the online model based on poisons from our adaptive attack
                    if len(online_poisons_y) > 0:
                        online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                        online_poisons_y = np.concatenate(online_poisons_y,axis=0)
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
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter = 1000)
                    model_p_online.fit(online_full_x, online_full_y) 
                    # need to save the posioned model from our attack, for the purpose of validating the lower bound for the online attack
                    if args.poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_online_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_online_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    joblib.dump(model_p_online, filename)
                    if args.poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    np.savez(filename,
                            online_poisons_x = online_poisons_x,
                            online_poisons_y = online_poisons_y,
                            best_lower_bound = best_lower_bound,
                            conser_lower_bound = conser_lower_bound,
                            best_max_loss_x = best_max_loss_x,
                            best_max_loss_y = best_max_loss_y,
                            target_poison_max_losses = target_poison_max_losses,
                            current_total_losses = current_total_losses, 
                            max_loss_diffs = max_loss_diffs_reg,
                            lower_bounds = lower_bounds,
                            ol_tol_params = ol_tol_params,
                            online_acc_scores = np.array(online_acc_scores),
                            norm_diffs = norm_diffs
                            ) 
            else:
                # load data
                if args.poison_whole:
                    if args.poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                data_info = np.load(filename)
                online_poisons_x = data_info["online_poisons_x"]
                online_poisons_y = data_info["online_poisons_y"]
                best_lower_bound = data_info["best_lower_bound"]
                conser_lower_bound = data_info["conser_lower_bound"]
                best_max_loss_x = data_info["best_max_loss_x"]
                best_max_loss_y = data_info["best_max_loss_y"]
                target_poison_max_losses = data_info["target_poison_max_losses"]
                current_total_losses = data_info["current_total_losses"]
                max_loss_diffs_reg = data_info["max_loss_diffs"]
                lower_bounds = data_info["lower_bounds"]
                ol_tol_params = data_info["ol_tol_params"] 
                ol_tol_par = ol_tol_params[-1] 
                online_acc_scores = data_info['online_acc_scores']
                norm_diffs = data_info['norm_diffs']
                # load model
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_online_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_online_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,target_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                model_p_online = joblib.load(open(filename, 'rb'))

            # summarize attack results
            final_norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                                X_train,
                                                                y_train,
                                                                X_test,
                                                                y_test,
                                                                subpop_data,
                                                                best_lower_bound,
                                                                conser_lower_bound,
                                                                kkt_tol_par,
                                                                ol_tol_par,
                                                                target_model,
                                                                kkt_model_p,
                                                                model_p_online,
                                                                len(target_online_poisons_y),
                                                                args)
            if best_lower_bound > len(target_online_poisons_y):
                print("violation observed for the lower bound of Adaptive attack model!")
                sys.exit(0)
            ol_target_lower_bound_and_attacks = ol_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,len(target_online_poisons_y),len(online_poisons_y),
                    kkt_tol_par, ol_tol_par] + final_norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores
            # write to csv files
            ol_lower_bound_writer.writerow(ol_target_lower_bound_and_attacks)
            
        if args.target_model == "compare":
            # only for the indiscriminate setting, compare our attack with improved target model generation and KKT attack
            # with original version of target model generation.
            if args.dataset == 'mnist_17':
                print("------- Compare our improved attack to baseline KKT attack -----")
                # load the kkt related model and data
                kkt_fraction = 1  
                speific_gen_proc = 'orig'
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                else:
                    filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_model_tol-{}_err-{}_kktfrac-{}.sav'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                kkt_model_p = joblib.load(open(filename, 'rb'))
                if args.poison_whole:
                    filename = 'files/kkt_models/{}/{}/{}/{}/whole-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                else:
                    filename = 'files/kkt_models/{}/{}/{}/{}/subpop-{}_data_tol-{}_err-{}_kktfrac-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err,kkt_fraction)
                data_info = np.load(filename)
                kkt_x_modified = data_info["kkt_x_modified"]
                kkt_y_modified = data_info["kkt_y_modified"]
                idx_poison = slice(X_train.shape[0], kkt_x_modified.shape[0])
                idx_clean = slice(0, X_train.shape[0])

                tst_target_acc = kkt_model_p.score(tst_sub_x, tst_sub_y)
                tst_collat_acc = kkt_model_p.score(tst_nsub_x, tst_nsub_y)
                print("--------Performance of Selected KKT attack model-------")
                # print('Total Train Acc : %.3f' % kkt_model_p.score(X_train, y_train))
                print('Total Test Acc : ', kkt_model_p.score(X_test, y_test))
                print('Test Target Acc : ', tst_target_acc)
                print('Test Collat Acc : ', tst_collat_acc)
                print('Total Train Acc : ', kkt_model_p.score(X_train, y_train))
                print('Train Target Acc : ', kkt_model_p.score(trn_sub_x, trn_sub_y))
                print('Train Collat Acc : ', kkt_model_p.score(trn_nsub_x,trn_nsub_y))

                # load the improved target model
                if args.poison_whole:
                    fname = open('files/target_classifiers/{}/improved_best_theta_whole_err-{}'.format(dataset_name,valid_theta_err), 'rb')  
                    
                else:
                    fname = open('files/target_classifiers/{}/improved_best_theta_subpop_{}_err-{}'.format(dataset_name,cl_ind,valid_theta_err), 'rb')  
                f = pickle.load(fname)
                tmp_best_target_theta = f['thetas']
                tmp_best_target_bias = f['biases']
                target_model.coef_= np.array([tmp_best_target_theta])
                target_model.intercept_ = np.array([tmp_best_target_bias])

                kkt_tol_par = 0
                speific_gen_proc = 'improved'
                # check if adaptive online attack is needed
                if args.poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                if not os.path.isfile(filename):
                    # start the lower bound computation process
                    # reset current model to clean model, just to measure the impact of curr_model
                    # curr_model.fit(X_train, y_train)
                    
                    print("[Sanity Compare] Acc of current model:",curr_model.score(X_test,y_test),curr_model.score(X_train,y_train))
                    args.fixed_budget = kkt_x_modified.shape[0]-X_train.shape[0]
                    print("Now run the online attack with fixed number of points:",args.fixed_budget)
                    assert args.fixed_budget > 0
                    

                    online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x,\
                    best_max_loss_y, ol_tol_par, target_poison_max_losses, current_total_losses,\
                    ol_tol_params, max_loss_diffs_reg, lower_bounds, online_acc_scores,norm_diffs = incre_online_learning(X_train,
                                                                                            y_train,
                                                                                            X_test,
                                                                                            y_test,
                                                                                            curr_model,
                                                                                            target_model,
                                                                                            x_lim_tuples,
                                                                                            args,
                                                                                            ScikitModel,
                                                                                            target_model_type = "real",
                                                                                            attack_num_poison = kkt_x_modified.shape[0]-X_train.shape[0],
                                                                                            kkt_tol_par = kkt_tol_par,
                                                                                            subpop_data = subpop_data,
                                                                                            target_poisons = poisons_all)
                    # retrain the online model based on poisons from our adaptive attack
                    if len(online_poisons_y) > 0:
                        online_poisons_x = np.concatenate(online_poisons_x,axis=0)
                        online_poisons_y = np.concatenate(online_poisons_y,axis=0)
                        online_full_x = np.concatenate((X_train,online_poisons_x),axis = 0)
                        print(y_train.shape,online_poisons_y.shape)
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
                        random_state=args.rand_seed,
                        verbose=False,
                        max_iter = 1000)
                    model_p_online.fit(online_full_x, online_full_y) 

                    # need to save the posioned model from our attack, for the purpose of validating the lower bound for the online attack
                    if args.poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_compare_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_compare_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    joblib.dump(model_p_online, filename)
                    if args.poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    np.savez(filename,
                            online_poisons_x = online_poisons_x,
                            online_poisons_y = online_poisons_y,
                            best_lower_bound = best_lower_bound,
                            conser_lower_bound = conser_lower_bound,
                            best_max_loss_x = best_max_loss_x,
                            best_max_loss_y = best_max_loss_y,
                            target_poison_max_losses = target_poison_max_losses,
                            current_total_losses = current_total_losses, 
                            max_loss_diffs = max_loss_diffs_reg,
                            lower_bounds = lower_bounds,
                            ol_tol_params = ol_tol_params,
                            online_acc_scores = np.array(online_acc_scores),
                            norm_diffs = norm_diffs
                            )
                else:
                    if args.poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    data_info = np.load(filename)
                    online_poisons_x = data_info["online_poisons_x"]
                    online_poisons_y = data_info["online_poisons_y"]
                    best_lower_bound = data_info["best_lower_bound"]
                    conser_lower_bound = data_info["conser_lower_bound"]
                    best_max_loss_x = data_info["best_max_loss_x"]
                    best_max_loss_y = data_info["best_max_loss_y"]
                    target_poison_max_losses = data_info["target_poison_max_losses"]
                    current_total_losses = data_info["current_total_losses"]
                    max_loss_diffs_reg = data_info["max_loss_diffs"]
                    lower_bounds = data_info["lower_bounds"]
                    ol_tol_params = data_info["ol_tol_params"]  
                    ol_tol_par = ol_tol_params[-1]
                    online_acc_scores = data_info['online_acc_scores']
                    norm_diffs = data_info['norm_diffs']

                    if args.poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/whole-{}_online_for_compare_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/subpop-{}_online_for_compare_model_tol-{}_err-{}.sav'.format(dataset_name,args.rand_seed,speific_gen_proc,args.repeat_num,cl_ind,args.incre_tol_par,valid_theta_err)
                    model_p_online = joblib.load(open(filename, 'rb'))

                # summarize the attack results
                final_norm_diffs,kkt_acc_scores, ol_acc_scores = compare_attack_and_lower_bound(online_poisons_y,
                                                                    X_train,
                                                                    y_train,
                                                                    X_test,
                                                                    y_test,
                                                                    subpop_data,
                                                                    best_lower_bound,
                                                                    conser_lower_bound,
                                                                    kkt_tol_par,
                                                                    ol_tol_par,
                                                                    target_model,
                                                                    kkt_model_p,
                                                                    model_p_online,
                                                                    kkt_x_modified.shape[0]-X_train.shape[0],
                                                                    args)
                compare_target_lower_bound_and_attacks = compare_target_lower_bound_and_attacks + [best_lower_bound,conser_lower_bound,\
                    kkt_x_modified.shape[0]-X_train.shape[0],len(online_poisons_y),
                kkt_tol_par, ol_tol_par] + final_norm_diffs + orig_model_acc_scores + target_model_acc_scores + kkt_acc_scores + ol_acc_scores
                # write key attack info to the csv files
                compare_lower_bound_writer.writerow(compare_target_lower_bound_and_attacks)
        
    # close all files
    if args.target_model == "all":
        kkt_lower_bound_file.flush()
        kkt_lower_bound_file.close()
        real_lower_bound_file.flush()
        real_lower_bound_file.close()
        ol_lower_bound_file.flush()
        ol_lower_bound_file.close()
        # compare_lower_bound_file.flush()
        # compare_lower_bound_file.close()
    elif args.target_model == "kkt":
        kkt_lower_bound_file.flush()
        kkt_lower_bound_file.close()
    elif args.target_model == "real":
        real_lower_bound_file.flush()
        real_lower_bound_file.close()
    elif args.target_model == "ol":
        ol_lower_bound_file.flush()
        ol_lower_bound_file.close()
    elif args.target_model == "compare":
        compare_lower_bound_file.flush()
        compare_lower_bound_file.close()

