from sklearn.datasets import make_classification
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, linear_model
from sklearn import cluster
import csv
import pickle
import sklearn

import cvxpy as cvx
idx_improved  = 0
x = np.array([1,2,3])

# KKT attack related modules
import kkt_attack
from upper_bounds import hinge_loss, hinge_grad, logistic_grad
from datasets import load_dataset

import data_utils as data
import argparse
import os
import sys

from sklearn.externals import joblib

def make_dirs(dataset_name):
    if not os.path.isdir('files/results/{}/approx_optimal_attack'.format(dataset_name)):
        os.makedirs('files/results/{}/approx_optimal_attack'.format(dataset_name))
    if not os.path.isdir('files/kkt_models/{}/approx_optimal_attack'.format(dataset_name)):
        os.makedirs('files/kkt_models/{}/approx_optimal_attack'.format(dataset_name))

    if not os.path.isdir('files/results/{}'.format(dataset_name)):
        os.makedirs('files/results/{}'.format(dataset_name))
    if not os.path.isdir('files/kkt_models/{}'.format(dataset_name)):
        os.makedirs('files/kkt_models/{}'.format(dataset_name))
    # if not os.path.isdir('files/kkt_models/{}/all_models/'.format(dataset_name)):
    #     os.makedirs('files/kkt_models/{}/all_models/'.format(dataset_name))
    if not os.path.isdir('files/online_models/{}'.format(dataset_name)):
        os.makedirs('files/online_models/{}'.format(dataset_name))
    if not os.path.isdir('files/target_classifiers/{}'.format(dataset_name)):
        os.makedirs('files/target_classifiers/{}'.format(dataset_name))



    if not os.path.isdir('files/online_models/{}/check_valid_thetas'.format(dataset_name)):
        os.makedirs('files/online_models/{}/check_valid_thetas'.format(dataset_name))

    if not os.path.isdir('files/results/{}/check_valid_thetas/'.format(dataset_name)):
        os.makedirs('files/results/{}/check_valid_thetas/'.format(dataset_name))

def svm_model(**kwargs):
    return svm.LinearSVC(loss='hinge', **kwargs)
def logistic_model(**kwargs):
    return linear_model.LogisticRegression(**kwargs)

################# begin definition of some functions ################
def dist_to_boundary(theta,bias,data):
    abs_vals = np.abs(np.dot(data,theta) + bias)
    return abs_vals/(np.linalg.norm(theta,ord = 2))

def calculate_loss(margins):
    # hige loss calculation
    losses = np.maximum(1-margins, 0)
    errs = (margins < 0) + 0.5 * (margins == 0)
    return np.sum(losses)/len(margins), np.sum(errs)/len(errs)

def cvx_dot(a,b):
    return cvx.sum_entries(cvx.mul_elemwise(a, b))

def compute_grad_norm_diff(target_theta,target_bias,total_epsilon,\
    X_train,y_train,x_poisons,y_poisons,args):
    clean_grad_at_target_theta, clean_bias_grad_at_target_theta = hinge_grad(target_theta,
                                                                        target_bias,
                                                                        X_train,
                                                                        y_train)
    poison_grad_at_target_theta, poison_bias_at_target_theta = hinge_grad(target_theta,
                                                                        target_bias,
                                                                        x_poisons,
                                                                        y_poisons)

    poison_grad_at_target_theta = poison_grad_at_target_theta * x_poisons.shape[0]/X_train.shape[0]
    poison_bias_at_target_theta = poison_bias_at_target_theta * x_poisons.shape[0]/X_train.shape[0]
    target_grad = clean_grad_at_target_theta + ((1 + total_epsilon) * args.weight_decay * target_theta)
    grad_norm_diff = np.linalg.norm(target_grad + poison_grad_at_target_theta)
    return grad_norm_diff


def search_max_loss_pt_contin(clean_model,poison_model,y,x_lim_tuple,args):
    theta_c = clean_model.coef_.reshape(-1)
    bias_c = clean_model.intercept_
    theta_p = poison_model.coef_.reshape(-1)
    bias_p = poison_model.intercept_
    x_min, x_max = x_lim_tuple

    print("x_min and x_max:",x_min,x_max)
    # cvx variables and params
    if args.dataset == "adult":
        # used for the binary constraints
        arr = np.array([0]*(theta_c.shape[0]))
        arr[4:12] = 1
        cvx_work_class = cvx.Parameter(theta_c.shape[0], value = arr)
        arr = np.array([0]*(theta_c.shape[0]))
        arr[12:27] = 1
        cvx_education = cvx.Parameter(theta_c.shape[0], value = arr)
        arr = np.array([0]*(theta_c.shape[0]))
        arr[27:33] = 1
        cvx_martial = cvx.Parameter(theta_c.shape[0], value = arr)
        arr = np.array([0]*(theta_c.shape[0]))
        arr[33:47] = 1
        cvx_occupation = cvx.Parameter(theta_c.shape[0], value = arr)
        arr = np.array([0]*(theta_c.shape[0]))
        arr[47:52] = 1
        cvx_relationship = cvx.Parameter(theta_c.shape[0], value = arr)
        arr = np.array([0]*(theta_c.shape[0]))
        arr[52:57] = 1
        cvx_race = cvx.Parameter(theta_c.shape[0], value = arr)

    cvx_x = cvx.Variable(theta_c.shape[0])
    
    cvx_theta_c = cvx.Parameter(theta_c.shape[0])
    cvx_bias_c = cvx.Parameter(1)
    cvx_theta_p = cvx.Parameter(theta_c.shape[0])
    cvx_bias_p = cvx.Parameter(1)
    # assign param values
    cvx_theta_c.value = theta_c
    cvx_bias_c.value = bias_c
    cvx_theta_p.value = theta_p
    cvx_bias_p.value = bias_p
    max_loss = -1
    # # cvx objective related definitions
    # case 1: !0 loss for clean model, 0 for poison model
    print("explore case 1:")
    cvx_loss = 1-y * (cvx_dot(cvx_theta_c,cvx_x) + cvx_bias_c) 
    cvx_constraints = [
        y * (cvx_dot(cvx_theta_c,cvx_x) + cvx_bias_c) <= 1,
        y * (cvx_dot(cvx_theta_p,cvx_x) + cvx_bias_p) >= 1
    ]
    if x_lim_tuple:
        print("x real values are constrained!")
        cvx_constraints.append(cvx_x >= x_min)
        cvx_constraints.append(cvx_x <= x_max)
    if args.dataset == 'adult':
        # binary featutre constraints: beacuse of one-hot encoding
        cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        aa = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 1 debugging Info:")
        print("labels:",y)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        # print(cvx_theta_c.value,cvx_bias_c.value)
        # print(cvx_theta_p.value,cvx_bias_p.value)
        # print(x_min,x_max)
        aa = 0
 
    print("optimal value found from optimization:",aa)
    # obtain the max loss and best point for poisoning
    if aa!=0:
        if max_loss < cvx_prob.value:
            max_loss = cvx_prob.value
            max_loss_x = np.array(cvx_x.value)
            print("max loss is changed to:",max_loss)
    else:
        max_loss = 0
        max_loss_x = 0

    print("explore case 2:")
    # case 2: !0 loss for clean model, !0 for poison model
    cvx_loss = y * (cvx_dot(cvx_theta_p-cvx_theta_c,cvx_x) + (cvx_bias_p - cvx_bias_c))
    cvx_constraints = [
        y * (cvx_dot(cvx_theta_c,cvx_x) + cvx_bias_c) <= 1,
        y * (cvx_dot(cvx_theta_p,cvx_x) + cvx_bias_p) <= 1
    ]
    if x_lim_tuple:
        cvx_constraints.append(cvx_x >= x_min)
        cvx_constraints.append(cvx_x <= x_max)

    if args.dataset == 'adult':
        # binary featutre constraints: beacuse of one-hot encoding
        cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
        cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        aa = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 2 debugging Info:")
        print("labels:",y)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        # print(cvx_theta_c.value,cvx_bias_c.value)
        # print(cvx_theta_p.value,cvx_bias_p.value)
        # print(x_min,x_max)

        aa = 0
    
    print("optimal value found from optimization:",aa)
    # obtain the max loss and best point for poisoning
    if aa!=0:
        if max_loss < cvx_prob.value:
            max_loss = cvx_prob.value
            max_loss_x = np.array(cvx_x.value)
            print("max loss is changed to:",max_loss)

    # case 4: 0 for both cases, do not need to calculate.

    # print(max_loss,max_loss_x)
    print(max_loss)
    return max_loss, max_loss_x

def search_max_loss_pt(clean_model,poison_model,y,x_lim_tuple,args):
    theta_c = clean_model.coef_.reshape(-1)
    bias_c = clean_model.intercept_
    theta_p = poison_model.coef_.reshape(-1)
    bias_p = poison_model.intercept_
    x_min, x_max = x_lim_tuple

    # print("*******model weights************")
    # print("clean model weights:",theta_c,bias_c)
    # print("poison model weights:",theta_p,bias_p)
    print("x_min and x_max:",x_min,x_max)
    # cvx variables and params
    if args.dataset == "adult":
        # used for the binary constraints
        arr = np.array([0]*(theta_c.shape[0]-4))
        arr[0:8] = 1
        cvx_work_class = cvx.Parameter(theta_c.shape[0]-4, value = arr)
        arr = np.array([0]*(theta_c.shape[0]-4))
        arr[8:23] = 1
        cvx_education = cvx.Parameter(theta_c.shape[0]-4, value = arr)
        arr = np.array([0]*(theta_c.shape[0]-4))
        arr[23:29] = 1
        cvx_martial = cvx.Parameter(theta_c.shape[0]-4, value = arr)
        arr = np.array([0]*(theta_c.shape[0]-4))
        arr[29:43] = 1
        cvx_occupation = cvx.Parameter(theta_c.shape[0]-4, value = arr)
        arr = np.array([0]*(theta_c.shape[0]-4))
        arr[43:48] = 1
        cvx_relationship = cvx.Parameter(theta_c.shape[0]-4, value = arr)
        arr = np.array([0]*(theta_c.shape[0]-4))
        arr[48:53] = 1
        cvx_race = cvx.Parameter(theta_c.shape[0]-4, value = arr)

        cvx_x_real = cvx.Variable(4)
        cvx_x_binary = cvx.Bool(theta_c.shape[0]-4)
        cvx_x = cvx.vstack(cvx_x_real,cvx_x_binary)

        # # used for the binary constraints
        # arr = np.array([0]*(theta_c.shape[0]))
        # arr[4:12] = 1
        # cvx_work_class = cvx.Parameter(theta_c.shape[0], value = arr)
        # arr = np.array([0]*(theta_c.shape[0]))
        # arr[12:27] = 1
        # cvx_education = cvx.Parameter(theta_c.shape[0], value = arr)
        # arr = np.array([0]*(theta_c.shape[0]))
        # arr[27:33] = 1
        # cvx_martial = cvx.Parameter(theta_c.shape[0], value = arr)
        # arr = np.array([0]*(theta_c.shape[0]))
        # arr[33:47] = 1
        # cvx_occupation = cvx.Parameter(theta_c.shape[0], value = arr)
        # arr = np.array([0]*(theta_c.shape[0]))
        # arr[47:52] = 1
        # cvx_relationship = cvx.Parameter(theta_c.shape[0], value = arr)
        # arr = np.array([0]*(theta_c.shape[0]))
        # arr[52:57] = 1
        # cvx_race = cvx.Parameter(theta_c.shape[0], value = arr)
        # cvx_x = cvx.Variable(theta_c.shape[0])
    else:
        cvx_x = cvx.Variable(theta_c.shape[0])
    
    cvx_theta_c = cvx.Parameter(theta_c.shape[0])
    cvx_bias_c = cvx.Parameter(1)
    cvx_theta_p = cvx.Parameter(theta_c.shape[0])
    cvx_bias_p = cvx.Parameter(1)
    # assign param values
    cvx_theta_c.value = theta_c
    cvx_bias_c.value = bias_c
    cvx_theta_p.value = theta_p
    cvx_bias_p.value = bias_p
    max_loss = -1
    # # cvx objective related definitions
    # case 1: !0 loss for clean model, 0 for poison model
    print("explore case 1:")
    cvx_loss = 1-y * (cvx_dot(cvx_theta_c,cvx_x) + cvx_bias_c) 
    cvx_constraints = [
        y * (cvx_dot(cvx_theta_c,cvx_x) + cvx_bias_c) <= 1,
        y * (cvx_dot(cvx_theta_p,cvx_x) + cvx_bias_p) >= 1
    ]
    if x_lim_tuple:
        if args.dataset == 'adult':
            cvx_constraints.append(cvx_x_real >= x_min)
            cvx_constraints.append(cvx_x_real <= x_max)
        # if args.dataset == 'adult':
        #     print("x real values are constrained!")
        #     cvx_constraints.append(cvx_x >= x_min)
        #     cvx_constraints.append(cvx_x <= x_max)
        else:
            cvx_constraints.append(cvx_x >= x_min)
            cvx_constraints.append(cvx_x <= x_max)

    if args.dataset == 'adult':
        # binary featutre constraints: beacuse of one-hot encoding
        cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_education, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_martial, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_race, cvx_x_binary) == 1)

        # cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    
    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        aa = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 1 debugging Info:")
        print("labels:",y)
        # print(cvx_theta_c.value,cvx_bias_c.value)
        # print(cvx_theta_p.value,cvx_bias_p.value)
        # print(x_min,x_max)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        aa = 0
 
    print("optimal value found from optimization:",aa)
    # obtain the max loss and best point for poisoning
    if aa!=0:
        if max_loss < cvx_prob.value:
            max_loss = cvx_prob.value
            max_loss_x = np.array(cvx_x.value)
            print("max loss is changed to:",max_loss)
    else:
        max_loss = 0
        max_loss_x = 0

    print("explore case 2:")
    # case 2: !0 loss for clean model, !0 for poison model
    cvx_loss = y * (cvx_dot(cvx_theta_p-cvx_theta_c,cvx_x) + (cvx_bias_p - cvx_bias_c))
    cvx_constraints = [
        y * (cvx_dot(cvx_theta_c,cvx_x) + cvx_bias_c) <= 1,
        y * (cvx_dot(cvx_theta_p,cvx_x) + cvx_bias_p) <= 1
    ]
    if x_lim_tuple:
        if args.dataset == 'adult':
            cvx_constraints.append(cvx_x_real >= x_min)
            cvx_constraints.append(cvx_x_real <= x_max)
        # if args.dataset == 'adult':
        #     print("x real values are constrained!")
        #     cvx_constraints.append(cvx_x >= x_min)
        #     cvx_constraints.append(cvx_x <= x_max)
        else:
            cvx_constraints.append(cvx_x >= x_min)
            cvx_constraints.append(cvx_x <= x_max)

    if args.dataset == 'adult':
        # binary featutre constraints: beacuse of one-hot encoding
        cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_education, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_martial, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x_binary) == 1)
        cvx_constraints.append(cvx_dot(cvx_race, cvx_x_binary) == 1)

        # cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
        # cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        aa = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 2 debugging Info:")
        print("labels:",y)
        # print(cvx_theta_c.value,cvx_bias_c.value)
        # print(cvx_theta_p.value,cvx_bias_p.value)
        # print(x_min,x_max)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        aa = 0
    
    print("optimal value found from optimization:",aa)
    # obtain the max loss and best point for poisoning
    if aa!=0:
        if max_loss < cvx_prob.value:
            max_loss = cvx_prob.value
            max_loss_x = np.array(cvx_x.value)
            print("max loss is changed to:",max_loss)

    # case 4: 0 for both cases, do not need to calculate.

    # print(max_loss,max_loss_x)
    print(max_loss)
    return max_loss, max_loss_x

def print_for_debug(X_train,
                    y_train,
                    curr_model,
                    target_model,
                    max_loss_x,
                    max_loss_y,
                    x_lim_tuples,
                    args
                    ):
    # # #################  print all related info to debug the problem ################
    # assert np.array_equal(X_train, kkt_x_modified[idx_clean,:])
    # assert np.array_equal(y_train,kkt_y_modified[idx_clean])

    margins = y_train*(X_train.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    target_clean_total_loss = np.sum(np.maximum(1-margins, 0))

    margins = y_train*(X_train.dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)
    current_clean_total_loss = np.sum(np.maximum(1-margins, 0))
    # # since we use kkt target currently, make sure the total loss on whole data set is also minimized for target model
    # margins = kkt_y_modified*(kkt_x_modified.dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)
    # print(margins.shape)
    # current_kkt_total_loss= np.sum(np.maximum(1-margins, 0))
    # margins = kkt_y_modified*(kkt_x_modified.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    # print(margins.shape)
    # target_kkt_total_loss = np.sum(np.maximum(1-margins, 0))

    # print the loss of orig and target model with clean train data
    print("----------- debug total model loss---------")
    # print("size of clean and poisoned train data:",X_train.shape[0],kkt_x_modified.shape[0])
    curr_reg = (np.linalg.norm(curr_model.coef_.reshape(-1))**2+(curr_model.intercept_)**2)/2
    target_reg = (np.linalg.norm(target_model.coef_.reshape(-1))**2+(target_model.intercept_)**2)/2
    reg_diff = curr_reg - target_reg

    print("train loss of current model with clean train data {}, Regularization Term {}, Their Sum {}".format(current_clean_total_loss,\
        curr_reg*X_train.shape[0]*args.weight_decay,current_clean_total_loss+curr_reg*X_train.shape[0] * args.weight_decay))
    print("train loss of target model with clean train data {}, Regularization Term {}, Their Sum {}".format(target_clean_total_loss,\
        target_reg*X_train.shape[0] * args.weight_decay,target_clean_total_loss+target_reg*X_train.shape[0] * args.weight_decay))
    print("total train loss difference:",target_clean_total_loss-current_clean_total_loss-X_train.shape[0] * args.weight_decay*reg_diff)

    # print("train loss of current model with KKT posioned train data:",current_kkt_total_loss)
    # print("train loss of target model with KKT poisoned train data:",target_kkt_total_loss)

    # assert np.array_equal(X_train, kkt_x_modified[idx_clean,:])
    # assert np.array_equal(y_train,kkt_y_modified[idx_clean])

    # after_margins = kkt_y_modified[idx_poison]*(kkt_x_modified[idx_poison,:].dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)
    # print(after_margins.shape)
    # current_kkt_poison_loss= np.sum(np.maximum(1-after_margins, 0))
    # after_margins = kkt_y_modified[idx_poison]*(kkt_x_modified[idx_poison,:].dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    # print(after_margins.shape)
    # target_kkt_poison_loss = np.sum(np.maximum(1-after_margins, 0))
    # print("train loss of current model with only posioned train data:",current_kkt_poison_loss)
    # print("train loss of target model with only poisoned train data:",target_kkt_poison_loss)  

    # print("average loss for kkt poison points:",(current_kkt_poison_loss-target_kkt_poison_loss+after_margins.shape[0] * args.weight_decay*reg_diff)/after_margins.shape[0]) 
    # assert np.array_equal(X_train, kkt_x_modified[idx_clean,:])
    # assert np.array_equal(y_train,kkt_y_modified[idx_clean])
    # print("repititiono of the KKT points counts:",unique_counts)

    print("----------debug: max loss points------------")
    # sanity check: what is the loss function of kkt attack and max loss point?
    # first check if you can produce same results with the loss calculation method
    # max_loss_x = np.squeeze(max_loss_x)
    margins = max_loss_y*(max_loss_x.dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)
    max_loss_clean = np.maximum(1-margins, 0)
    margins = max_loss_y*(max_loss_x.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    max_loss_poison = np.maximum(1-margins, 0)
    print("label of max loss point:",max_loss_y)
    print("loss of max loss point for clean model {}, for poison model {}, their loss difference {}, reg difference {}".format(max_loss_clean,\
        max_loss_poison,max_loss_clean - max_loss_poison,reg_diff))
    print("the exact max loss difference (with regularization):",max_loss_clean - max_loss_poison + args.weight_decay*reg_diff)
    # max_loss_diff = max_loss_clean-max_loss_poison

    # print("----------debug: KKT points------------")
    # print("shape of unique KKT points:",kkt_unique_x.shape)
    # for iii in range(len(kkt_unique_x)):
    #     kkt_x = kkt_unique_x[iii]
    #     kkt_y = kkt_unique_y[iii]
    #     # print(kkt_x,kkt_y)
    #     # print(curr_model.coef_.reshape(-1),curr_model.intercept_)
    #     # print(target_model.coef_.reshape(-1),target_model.intercept_)
    #     margins = kkt_y*(kkt_x.dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)
    #     kkt_loss_curr = np.maximum(1-margins, 0)
    #     margins = kkt_y*(kkt_x.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    #     kkt_loss_target = np.maximum(1-margins, 0)
    #     if type(kkt_loss_curr) != np.float64:
    #         kkt_loss_curr = kkt_loss_curr[0]
    #     if type(kkt_loss_target) != np.float64:
    #         # print(kkt_loss_target)
    #         # print(type(kkt_loss_target))
    #         kkt_loss_target = kkt_loss_target[0]
    #     # print("kkt point and labels",kkt_x,kkt_y)
    #     print("kkt point labels",kkt_y)
    #     print("loss of kkt point on current and target models:",kkt_loss_curr,kkt_loss_target)
    #     print("the loss difference of kkt point, reg term and total difference is:",kkt_loss_curr-kkt_loss_target,\
    #         args.weight_decay*reg_diff,kkt_loss_curr-kkt_loss_target + args.weight_decay*reg_diff)
    #     margins = y_train*(X_train.dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)

    #     clean_total_loss = np.sum(np.maximum(1-margins, 0))
    #     margins = y_train*(X_train.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    #     poison_total_loss = np.sum(np.maximum(1-margins, 0))
    #     # print("total train loss difference:",poison_total_loss-clean_total_loss)
    #     if (kkt_loss_curr-kkt_loss_target) >0:
    #         kkt_lower_bound = int((poison_total_loss-clean_total_loss-X_train.shape[0] * args.weight_decay*reg_diff)/(kkt_loss_curr-kkt_loss_target+ args.weight_decay*reg_diff))
    #     else:
    #         print("kkt loss for current point is zero and set lower bound to 0")
    #         kkt_lower_bound = 0
    #     print("kkt lower bound:",kkt_lower_bound)
        
    #     # make sure the kkt point is in valid range
    #     if kkt_y == 1:
    #         x_pos_min, x_pos_max = x_lim_tuples[0]
    #         x_neg_min, x_neg_max = x_lim_tuples[1]
    #         assert (kkt_x >= x_pos_min - np.abs(float(x_pos_min))/100).all(), "x_pos_min {},x_pos_max {},x_neg_min {},x_neg_max {}".format(x_pos_min,x_pos_max,x_neg_min,x_neg_max) 
    #         assert (kkt_x <= x_pos_max + float(x_pos_max)/100).all(), "x_pos_min {},x_pos_max {},x_neg_min {},x_neg_max {}".format(x_pos_min,x_pos_max,x_neg_min,x_neg_max) 
    #     else:
    #         x_pos_min, x_pos_max = x_lim_tuples[0]
    #         x_neg_min, x_neg_max = x_lim_tuples[1]
    #         assert (kkt_x >= x_neg_min - np.abs(float(x_neg_min))/100).all(), "x_pos_min {},x_pos_max {},x_neg_min {},x_neg_max {}".format(x_pos_min,x_pos_max,x_neg_min,x_neg_max) 
    #         assert (kkt_x <= x_neg_max + float(x_neg_max)/100).all(), "x_pos_min {},x_pos_max {},x_neg_min {},x_neg_max {}".format(x_pos_min,x_pos_max,x_neg_min,x_neg_max) 

    return max_loss_poison
    ########### end of debugging the info ############ 

def incre_online_learning(X_train,
                        y_train,
                        curr_model,
                        target_model,
                        x_lim_tuples,
                        args,
                        ScikitModel,
                        target_model_type,
                        attack_num_poison,
                        kkt_tol_par):

    num_iter = 0
    best_lower_bound = 0
    conser_lower_bound = 0
    online_poisons_x = []
    online_poisons_y = []

    theta_ol = curr_model.coef_
    bias_ol = curr_model.intercept_

    # loss of target model
    margins = y_train*(X_train.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    print(margins.shape)
    target_total_loss = np.sum(np.maximum(1-margins, 0))
    # Search the current max loss difference point for initial model pairs
    classes = [-1,1]
    best_loss = -1  
    for cls1 in classes:
        if cls1 == -1:
            max_loss, max_x = search_max_loss_pt(curr_model,target_model,cls1,x_lim_tuples[1],args)
            if best_loss < max_loss:
                best_loss = max_loss
                max_loss_x = max_x
                max_loss_y = -1
        else:
            max_loss, max_x = search_max_loss_pt(curr_model,target_model,cls1,x_lim_tuples[0],args)
            if best_loss < max_loss:
                best_loss = max_loss
                max_loss_x = max_x
                max_loss_y = 1
    # compute the conservative lower bound using relaxation for integer programming
    if args.dataset == 'adult':
        best_loss_real = -1 
        for cls1 in classes:
            if cls1 == -1:
                max_loss_real, _ = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[1],args)
                if best_loss_real < max_loss_real:
                    best_loss_real = max_loss_real
            else:
                max_loss_real, _ = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[0],args)
                if best_loss_real < max_loss_real:
                    best_loss_real = max_loss_real

    if args.online_alg_criteria == "max_loss":
        current_tol_par = best_loss
    else:
        current_tol_par = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-theta_ol.reshape(-1))**2+(target_model.intercept_ - bias_ol[0])**2)
    print("tolerance parameter of initial model and target model:",current_tol_par)
    # while current_tol_par > 1e-3 and num_iter <= 0.5*X_train.shape[0]:
    tmp_x = np.copy(X_train)
    tmp_y= np.copy(y_train)
    lower_bound = 0
    # set the stop criteria for online learning algorithm
    if target_model_type == "compare":
        ol_lr_threshold = kkt_tol_par
        print("compare the adaptive and kkt attack under same stop croteria!")
    else:
        ol_lr_threshold = args.incre_tol_par
    print("Stop criteria of adaptive poisoning attack:",ol_lr_threshold)
    target_poison_max_losses = []
    max_loss_diffs_reg = []
    ol_tol_params = []
    total_loss_diffs = []
    lower_bounds = []
    # print(current_tol_par, ol_lr_threshold, current_tol_par - ol_lr_threshold)
    # assert current_tol_par > ol_lr_threshold
    while current_tol_par > ol_lr_threshold:
        num_iter += 1
        # # set the model weights to proper value
        # curr_model.coef_ = theta_ol
        # curr_model.intercept_ = bias_ol
    
        # compute all the loss and then get the minimum amount of poisons
        margins = y_train*(X_train.dot(theta_ol.reshape(-1)) + bias_ol)
        current_total_loss = np.sum(np.maximum(1-margins, 0))
        print("the loss difference of current and target model:",target_total_loss - current_total_loss)

        # for original and new models, compute theoretical lower bound
        assert np.array_equal(theta_ol,curr_model.coef_)
        assert bias_ol == curr_model.intercept_

        max_loss_x = np.transpose(max_loss_x)
        # print(max_loss_x,max_loss_x.shape)

        # calibrated version of computing lower bound
        curr_reg = (np.linalg.norm(curr_model.coef_.reshape(-1))**2+(curr_model.intercept_)**2)/2
        target_reg = (np.linalg.norm(target_model.coef_.reshape(-1))**2+(target_model.intercept_)**2)/2
        reg_diff = curr_reg - target_reg
        print("regularizer of current model {}, regularizer of target model {}, reg diff: {}".format(curr_reg,target_reg,reg_diff))

        calib_max_loss_diff = best_loss + args.weight_decay *reg_diff
        total_loss_diff = target_total_loss - current_total_loss - args.weight_decay*X_train.shape[0]*reg_diff
        if args.dataset == 'adult':
            calib_max_loss_diff_real = best_loss_real + args.weight_decay *reg_diff
        # if target_total_loss < current_total_loss or target_total_loss_kkt > current_total_loss_kkt or best_loss <= 0:
        if total_loss_diff < 0:
            print("Total loss difference is negaative, don't update lower bound!")
        elif calib_max_loss_diff < 0:
            print("Best max loss point is negative loss, amazing. don't update lower bound!")
        else:
            # print("total train loss difference:",target_total_loss - current_total_loss)
            # lower_bound = int((target_total_loss - current_total_loss)/best_loss)
            print("total train loss difference (Reg Included) and max loss diff:",total_loss_diff, calib_max_loss_diff)
            lower_bound = int((total_loss_diff)/(calib_max_loss_diff))
            if args.dataset == 'adult':
                lower_bound_real = int((total_loss_diff)/(calib_max_loss_diff_real))
            else:
                lower_bound_real = lower_bound
            print("the computed lower bound is:",lower_bound)
            print("the conservative lower bound is:",lower_bound_real)
        
        # record the total loss diffs and max loss diffs
        total_loss_diffs.append(total_loss_diff)
        max_loss_diffs_reg.append(calib_max_loss_diff)
        ol_tol_params.append(current_tol_par)

        # append the lower bound w.r.t. iterations 
        lower_bounds.append(lower_bound)

        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound
            best_max_loss_x = max_loss_x
            best_max_loss_y = max_loss_y
        if lower_bound_real > conser_lower_bound:
            conser_lower_bound = lower_bound_real
        x_poisons = max_loss_x
        y_poisons = np.array([max_loss_y])
        
        max_loss_poison = print_for_debug(
                        X_train,
                        y_train,
                        curr_model,
                        target_model,
                        max_loss_x,
                        max_loss_y,
                        x_lim_tuples,
                        args)

        if target_model_type in ["kkt","ol"]:
            if lower_bound > attack_num_poison:
                print("something wrong with lower bound for online attack or KKT model!")
                sys.exit(0)

        ### update the model weights accordingly
        # curr_model.fit(np.concatenate((X_train,x_poisons),axis=0),np.concatenate((y_train,y_poisons),axis=0))
        print("size of tmp_x and x_poisons",tmp_x.shape,x_poisons.shape)
        tmp_x = np.concatenate((tmp_x,x_poisons),axis=0)
        tmp_y = np.concatenate((tmp_y,y_poisons),axis=0)
        print("the updated train set size is:",tmp_x.shape,tmp_y.shape)

        # refit the current model, now the update rule is 
        # curr_model.fit(tmp_x,tmp_y)
        C = 1.0 / (tmp_x.shape[0] * args.weight_decay)
        # train unpoisoned model
        fit_intercept = True
        curr_model = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 1000)
        curr_model.fit(tmp_x, tmp_y)
        theta_ol = curr_model.coef_
        bias_ol = curr_model.intercept_
        # update the real poison set
        online_poisons_x.append(max_loss_x)
        online_poisons_y.append(max_loss_y)

        # search the max loss difference point with updated model pair
        best_loss = -1 
        for cls1 in classes:
            if cls1 == -1:
                max_loss, max_x = search_max_loss_pt(curr_model,target_model,cls1,x_lim_tuples[1],args)
                if best_loss < max_loss:
                    best_loss = max_loss
                    max_loss_x = max_x
                    max_loss_y = -1
            else:
                max_loss, max_x = search_max_loss_pt(curr_model,target_model,cls1,x_lim_tuples[0],args)
                if best_loss < max_loss:
                    best_loss = max_loss
                    max_loss_x = max_x
                    max_loss_y = 1

        # for adult dataset, compute the relaxed version of max loss point and give conservative lower bound
        if args.dataset == 'adult':
            print("Use relaxed version for max loss point search!")
            best_loss_real = -1 
            for cls1 in classes:
                if cls1 == -1:
                    max_loss_real, _ = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[1],args)
                    if best_loss_real < max_loss_real:
                        best_loss_real = max_loss_real
                else:
                    max_loss_real, _ = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[0],args)
                    if best_loss_real < max_loss_real:
                        best_loss_real = max_loss_real
                    
        if args.online_alg_criteria == "max_loss":
            current_tol_par = best_loss
        else:
            current_tol_par = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-theta_ol.reshape(-1))**2+(target_model.intercept_ - bias_ol[0])**2)
            
        # current_tol_par = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-theta_ol.reshape(-1))**2+(target_model.intercept_ - bias_ol)**2)
        print("current telerance parameter is:",current_tol_par)
        # append the max loss point of max_loss point found w.r.t iterations
        target_poison_max_losses.append(max_loss_poison) 

    # finish up the last time info of the adaptive attack
    margins = y_train*(X_train.dot(theta_ol.reshape(-1)) + bias_ol)
    current_total_loss = np.sum(np.maximum(1-margins, 0))
    # calibrated version of computing lower bound
    curr_reg = (np.linalg.norm(curr_model.coef_.reshape(-1))**2+(curr_model.intercept_)**2)/2
    target_reg = (np.linalg.norm(target_model.coef_.reshape(-1))**2+(target_model.intercept_)**2)/2
    reg_diff = curr_reg - target_reg
    print("[Final] regularizer of current model {}, regularizer of target model {}, reg diff: {}".format(curr_reg,target_reg,reg_diff))

    calib_max_loss_diff = best_loss + args.weight_decay *reg_diff
    total_loss_diff = target_total_loss - current_total_loss - args.weight_decay*X_train.shape[0]*reg_diff
    total_loss_diffs.append(total_loss_diff)
    print("[Final] the pure loss difference of current and target model:",target_total_loss - current_total_loss)
    max_loss_diffs_reg.append(calib_max_loss_diff)
    ol_tol_params.append(current_tol_par)

    # squeeze for proper shape
    target_poison_max_losses = np.array(target_poison_max_losses)
    total_loss_diffs = np.array(total_loss_diffs)
    ol_tol_params = np.array(ol_tol_params)
    max_loss_diffs_reg = np.array(max_loss_diffs_reg)
    lower_bounds = np.array(lower_bounds)
    return online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x, best_max_loss_y,\
         current_tol_par, np.squeeze(target_poison_max_losses), np.squeeze(total_loss_diffs), np.squeeze(ol_tol_params),\
              np.squeeze(max_loss_diffs_reg), np.squeeze(lower_bounds)

def compare_attack_and_lower_bound(online_poisons_y,
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
                                   kkt_num_poison,
                                   args
                                   ):

    trn_sub_x,trn_sub_y,trn_nsub_x,trn_nsub_y,\
        tst_sub_x,tst_sub_y,tst_nsub_x,tst_nsub_y = subpop_data

    # print the lower bound and performance of different attacks 
    print("certified lower bound is:",best_lower_bound)
    print("conservative lower bound is:",conser_lower_bound)

    print("------performance of KKT attack-------")
    print("criteria difference to target classifier:",kkt_tol_par)
    target_model_b = target_model.intercept_
    target_model_b = target_model_b[0]
    kkt_model_p_b = kkt_model_p.intercept_
    kkt_model_p_b = kkt_model_p_b[0]
    kkt_norm_diff = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-kkt_model_p.coef_.reshape(-1))**2+(target_model_b - kkt_model_p_b)**2)
    print("norm difference to target classifier:",kkt_norm_diff)
    print("number of poisons:",kkt_num_poison)
    print()
    total_tst_acc = kkt_model_p.score(X_test, y_test)
    target_tst_acc = kkt_model_p.score(tst_sub_x, tst_sub_y)
    collat_tst_acc = kkt_model_p.score(tst_nsub_x,tst_nsub_y)

    total_trn_acc = kkt_model_p.score(X_train, y_train)
    target_trn_acc = kkt_model_p.score(trn_sub_x, trn_sub_y)
    collat_trn_acc = kkt_model_p.score(trn_nsub_x,trn_nsub_y)
    print('Total Test Acc: %.3f' % total_tst_acc)
    print('Test Target Acc : %.3f' % target_tst_acc)
    print('Test Collat Acc : %.3f' % collat_tst_acc)

    print('Total Train Acc: %.3f' % total_trn_acc)
    print('Train Target Acc : %.3f' % target_trn_acc)
    print('Train Collat Acc : %.3f' % collat_trn_acc)
    kkt_acc_scores = [total_tst_acc,target_tst_acc,collat_tst_acc,total_trn_acc,target_trn_acc,collat_trn_acc]

    print("------performance of Online attack-------")
    print("criteria difference to target classifier:",ol_tol_par)
    target_model_b = target_model.intercept_
    target_model_b = target_model_b[0]
    model_p_online_b = model_p_online.intercept_
    model_p_online_b = model_p_online_b[0]
    ol_norm_diff = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-model_p_online.coef_.reshape(-1))**2+(target_model_b - model_p_online_b)**2)
    print("norm difference to target classifier:",ol_norm_diff)
    
    print("number of poisons:",len(online_poisons_y))
    print()
    total_tst_acc = model_p_online.score(X_test, y_test)
    target_tst_acc = model_p_online.score(tst_sub_x, tst_sub_y)
    collat_tst_acc = model_p_online.score(tst_nsub_x,tst_nsub_y)

    total_trn_acc = model_p_online.score(X_train, y_train)
    target_trn_acc = model_p_online.score(trn_sub_x, trn_sub_y)
    collat_trn_acc = model_p_online.score(trn_nsub_x,trn_nsub_y)
    print('Total Test Acc: %.3f' % total_tst_acc)
    print('Test Target Acc : %.3f' % target_tst_acc)
    print('Test Collat Acc : %.3f' % collat_tst_acc)

    print('Total Train Acc: %.3f' % total_trn_acc)
    print('Train Target Acc : %.3f' % target_trn_acc)
    print('Train Collat Acc : %.3f' % collat_trn_acc)
    ol_acc_scores = [total_tst_acc,target_tst_acc,collat_tst_acc,total_trn_acc,target_trn_acc,collat_trn_acc]
    return [kkt_norm_diff, ol_norm_diff],kkt_acc_scores,ol_acc_scores
