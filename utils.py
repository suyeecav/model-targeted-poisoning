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

def random_sample(low,high):
    return (high-low) * np.random.random_sample() + low
def make_dirs(args):
    dataset_name = args.dataset
    if args.improved:
        tar_gen_proc = 'improved'
    else:
        tar_gen_proc = 'orig'
    rand_seed = args.rand_seed

    if not os.path.isdir('files/results/{}/{}/{}/{}/{}/approx_optimal_attack'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num)):
        os.makedirs('files/results/{}/{}/{}/{}/{}/approx_optimal_attack'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num))
    if not os.path.isdir('files/kkt_models/{}/{}/{}/{}/{}/approx_optimal_attack'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num)):
        os.makedirs('files/kkt_models/{}/{}/{}/{}/{}/approx_optimal_attack'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num))

    if not os.path.isdir('files/results/{}/{}/{}/{}/{}'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num)):
        os.makedirs('files/results/{}/{}/{}/{}/{}'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num))
    if not os.path.isdir('files/kkt_models/{}/{}/{}/{}/{}'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num)):
        os.makedirs('files/kkt_models/{}/{}/{}/{}/{}'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num))
    # if not os.path.isdir('files/kkt_models/{}/all_models/'.format(dataset_name)):
    #     os.makedirs('files/kkt_models/{}/all_models/'.format(dataset_name))
    if not os.path.isdir('files/online_models/{}/{}/{}/{}/{}'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num)):
        os.makedirs('files/online_models/{}/{}/{}/{}/{}'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num))
    if not os.path.isdir('files/target_classifiers/{}/{}'.format(dataset_name,args.model_type)):
        os.makedirs('files/target_classifiers/{}/{}'.format(dataset_name,args.model_type))

    if not os.path.isdir('files/online_models/{}/{}/{}/{}/{}/check_valid_thetas'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num)):
        os.makedirs('files/online_models/{}/{}/{}/{}/{}/check_valid_thetas'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num))

    if not os.path.isdir('files/results/{}/{}/{}/{}/{}/check_valid_thetas/'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num)):
        os.makedirs('files/results/{}/{}/{}/{}/{}/check_valid_thetas/'.format(dataset_name,args.model_type,rand_seed,tar_gen_proc,args.repeat_num))

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
    # computes the norm of the difference of gradient,
    # details can be found in the objective function of KKT attack
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

# below is related to logistic regression loss functions
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))
def log_likelihood(features, target, weights,bias):
    scores = np.dot(features, weights) + bias
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll
def compute_max_loss_diff(x,y,theta_c,bias_c,theta_p,bias_p):
    neg_ll_c = -log_likelihood(x, y, theta_c,bias_c)
    neg_ll_p = -log_likelihood(x, y, theta_p,bias_p)
    return neg_ll_c - neg_ll_p

def lr_search_max_loss_pt(d,curr_model,target_model,y,x_lim_tuple,args,lr=1e-5,num_steps=3000,trials=10,optimizer = 'adam'):
    # deply gradient descend strategy to search for apprximately max loss
    # optimizer: 'gd': gradent descend; 'adagrad', 'adam'; empirically, adam seems to converge much faster than the other two
    print("--- Testing with label ----:",y)
    # point for logistic regression
    if y == -1:
        y_tmp = 0
    else:
        y_tmp = y
    
    # for reproducibility
    np.random.seed(args.rand_seed)

    theta_c = curr_model.coef_.reshape(-1)
    bias_c = curr_model.intercept_
    theta_p = target_model.coef_.reshape(-1)
    bias_p = target_model.intercept_
    x_min, x_max = x_lim_tuple

    # setup the initial point for optimization and gradients
    # note: ll = np.sum( y*prediction - np.log(1 + np.exp(prediction)) ) 
    
    # x = np.zeros(d)
    # x1 = np.zeros(d)
    # print("before: min max",np.amin(x),np.amax(x))
    best_loss = -1e10
    # best_loss1 = -1e10
    for trial in range(trials):
        # print("------ trial {}------".format(trial))
        if args.dataset == 'dogfish':
            x = np.array([random_sample(x_min[i],x_max[i]) for i in range(len(x_min))])
            # x1 = np.array([random_sample(x_min[i],x_max[i]) for i in range(len(x_min))])
            # print(x.shape,x1.shape)
            # print(np.amax(x),np.amin(x))
            # print(np.amax(x1),np.amin(x1))
        else:
            x = np.array([random_sample(x_min,x_max) for i in range(d)])
            # x1 = np.array([random_sample(x_min,x_max) for i in range(d)])

        if optimizer == 'adagrad':
            # store the square of gradients
            # print("Utilizing Adagrad Optimizer")
            grads_squared = np.zeros(d)
            initial_accumulator_value = 0.001
            grads_squared.fill(initial_accumulator_value)
            epsilon = 1e-7
        elif optimizer == 'adam':
            # print("Utilizing Adam Optimizer")
            grads_first_moment = np.zeros(d)
            grads_second_moment = np.zeros(d)
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
        
        prev_loss = 1e10
        for step in range(num_steps):
            # if step == 0:
            #     max_loss = compute_max_loss_diff(x, y_tmp, theta_c,bias_c,theta_p,bias_p)
            #     print("(random) initial max loss value:",max_loss)
            # predictions of current and target models
            scores = np.dot(theta_c, x) + bias_c
            prediction_c = sigmoid(scores[0])
            scores = np.dot(theta_p, x) + bias_p
            prediction_p = sigmoid(scores[0])  

            # Update weights with gradient
            output_error_signal_c = prediction_c - y_tmp 
            output_error_signal_p = prediction_p - y_tmp 

            # the gradient is with respect to negative log likelihood
            # print(output_error_signal_c,theta_c.shape,x.shape)
            gradient_c = np.dot(theta_c, output_error_signal_c)
            gradient_p = np.dot(theta_p, output_error_signal_p)
            grads = (gradient_c - gradient_p)
            if optimizer == 'gd':
                x += lr * grads
            elif optimizer == 'adagrad':
                """Weights update using adagrad.
                grads2 = grads2 + grads**2
                w' = w - lr * grads / (sqrt(grads2) + epsilon)
                """
                grads_squared = grads_squared + grads**2
                x = x + lr * grads / (np.sqrt(grads_squared) + epsilon)
            elif optimizer == 'adam':
                """Weights update using Adam.
                
                g1 = beta1 * g1 + (1 - beta1) * grads
                g2 = beta2 * g2 + (1 - beta2) * g2
                g1_unbiased = g1 / (1 - beta1**time)
                g2_unbiased = g2 / (1 - beta2**time)
                w = w - lr * g1_unbiased / (sqrt(g2_unbiased) + epsilon)
                """
                time = step + 1
                grads_first_moment = beta1 * grads_first_moment + \
                                        (1. - beta1) * grads
                grads_second_moment = beta2 * grads_second_moment + \
                                        (1. - beta2) * grads**2
                
                grads_first_moment_unbiased = grads_first_moment / (1. - beta1**time)
                grads_second_moment_unbiased = grads_second_moment / (1. - beta2**time)
                
                x = x + lr * grads_first_moment_unbiased /(np.sqrt(grads_second_moment_unbiased) + epsilon)
            # print(y_tmp,output_error_signal_c, output_error_signal_p)
            # projection step to ensure it is within bounded norm
            x = np.clip(x,x_min,x_max)
            
            # print("added: min max",np.amin(lr * (gradient_c - gradient_p)),np.amax(lr * (gradient_c - gradient_p)))
            # print("before: min max",np.amin(x),np.amax(x))

            # max loss found so far
            if args.dataset == 'adult':
                # round the continuous values into discrete one to ensure it's meaningful
                x_tmp = np.copy(x)
                x_tmp[4:57] = np.rint(x[4:57]) 
                max_loss = compute_max_loss_diff(x_tmp, y_tmp, theta_c,bias_c,theta_p,bias_p)
                max_loss_real = compute_max_loss_diff(x, y_tmp, theta_c,bias_c,theta_p,bias_p)
            else:
                max_loss = compute_max_loss_diff(x, y_tmp, theta_c,bias_c,theta_p,bias_p)
                max_loss_real = max_loss
            
            if best_loss < max_loss:
                best_loss = max_loss
                best_loss_real = max_loss_real
                if args.dataset == 'adult':
                    best_x = x_tmp
                else:
                    best_x = x

            if np.abs(prev_loss - max_loss) < 1e-7:
                # print("Enough convergence")
                # print("steps: {}  max loss: {:.4f}  best_loss: {:.4f}".format(step+1, max_loss, best_loss))
    
                break

            prev_loss = max_loss
            # # also compute from target to curr to verify
            # scores1 = np.dot(theta_c, x1) + bias_c
            # prediction_c1 = sigmoid(scores1[0])
            # scores1 = np.dot(theta_p, x1) + bias_p
            # prediction_p1 = sigmoid(scores1[0])  

            # output_error_signal_c1 = prediction_c1 - y_tmp
            # output_error_signal_p1 = prediction_p1 - y_tmp

            # gradient_c1 = np.dot(theta_c, output_error_signal_c1)
            # gradient_p1 = np.dot(theta_p, output_error_signal_p1)

            # x1 += lr * (gradient_p1 - gradient_c1)
            # x1 = np.clip(x1,x_min,x_max)
            # max_loss1 = compute_max_loss_diff(x1, y_tmp, theta_p,bias_p,theta_c,bias_c)
            # if best_loss1 < max_loss1:
            #     best_loss1 = max_loss1

            # # Print log-likelihood every so often
            # if (step+1) % 1000 == 0:
            #     print("curr to target:",max_loss)
                # print("target to curr:",max_loss1)

        # print("selected best loss:",best_loss)
        # print("selected best loss1:",best_loss1)

    print("selected max loss with label {}: {}".format(y,best_loss))
    # print(best_x)
    
    return best_loss, best_loss_real, np.transpose(np.array([best_x]))

def search_max_loss_pt_contin(clean_model,poison_model,y,x_lim_tuple,args):
    theta_c = clean_model.coef_.reshape(-1)
    bias_c = clean_model.intercept_
    theta_p = poison_model.coef_.reshape(-1)
    bias_p = poison_model.intercept_
    x_min, x_max = x_lim_tuple
    if args.dataset != "dogfish": 
        print("x_min and x_max:",x_min,x_max)
    # cvx variables and params
    if args.dataset == "adult":
        # used for the binary constraints, however, the constraints are not used here
        # because the original data violates these constraints 
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
    
    # original Adult dataset does not obey the rules and remove following constraints for fair comparison 
    # if args.dataset == 'adult':
    #     # binary featutre constraints: beacuse of one-hot encoding
    #     cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        tmp_sol = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 1 debugging Info:")
        print("labels:",y)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        tmp_sol = 0
 
    print("optimal value found from optimization:",tmp_sol)
    # obtain the max loss and best point for poisoning
    if tmp_sol!=0:
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

    # if args.dataset == 'adult':
    #     # binary featutre constraints: beacuse of one-hot encoding
    #     cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) <= 1)
    #     cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        tmp_sol = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 2 debugging Info:")
        print("labels:",y)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        tmp_sol = 0
    
    print("optimal value found from optimization:",tmp_sol)
    # obtain the max loss and best point for poisoning
    if tmp_sol!=0:
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

    if args.dataset != "dogfish":
        print("min and max values of datapoint",x_min,x_max)

    # print("*******model weights************")
    # print("clean model weights:",theta_c,bias_c)
    # print("poison model weights:",theta_p,bias_p)

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
        else:
            cvx_constraints.append(cvx_x >= x_min)
            cvx_constraints.append(cvx_x <= x_max)

    # # original Adult data does not strictly obey the following constraints 
    # # and we comment them for fair comparison
    # if args.dataset == 'adult':
    #     # binary featutre constraints: beacuse of one-hot encoding
    #     cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x_binary) <= 1)
    #     cvx_constraints.append(cvx_dot(cvx_education, cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_martial, cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_race, cvx_x_binary) == 1)

    #     # cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        tmp_sol = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 1 debugging Info:")
        print("labels:",y)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        tmp_sol = 0
 
    print("optimal value found from optimization:",tmp_sol)
    # obtain the max loss and best point for poisoning
    if tmp_sol!=0:
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
        else:
            cvx_constraints.append(cvx_x >= x_min)
            cvx_constraints.append(cvx_x <= x_max)
    
    # # commented below also because original data violates the 
    # # strict constramts below
    # if args.dataset == 'adult':
    #     # binary featutre constraints: beacuse of one-hot encoding
    #     cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_education, cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_martial, cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x_binary) == 1)
    #     cvx_constraints.append(cvx_dot(cvx_race, cvx_x_binary) == 1)

    #     # cvx_constraints.append(cvx_dot(cvx_work_class, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_education, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_martial, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_occupation , cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_relationship, cvx_x) == 1)
    #     # cvx_constraints.append(cvx_dot(cvx_race, cvx_x) == 1)

    cvx_objective = cvx.Maximize(cvx_loss)
    cvx_prob = cvx.Problem(cvx_objective,cvx_constraints)
    try:
        tmp_sol = cvx_prob.solve(verbose=False, solver=cvx.GUROBI) 
    except cvx.error.SolverError:
        print("Case 2 debugging Info:")
        print("labels:",y)
        norm_diff = np.sqrt(np.linalg.norm(cvx_theta_c.value-cvx_theta_p.value)**2+(cvx_bias_c.value - cvx_bias_p.value)**2)
        print("norm difference:",norm_diff)
        tmp_sol = 0
    
    print("optimal value found from optimization:",tmp_sol)
    # obtain the max loss and best point for poisoning
    if tmp_sol!=0:
        if max_loss < cvx_prob.value:
            max_loss = cvx_prob.value
            max_loss_x = np.array(cvx_x.value)
            print("max loss is changed to:",max_loss)

    # case 4: 0 for both cases, do not need to calculate.
    print(max_loss)
    if args.dataset != "dogfish":
        assert np.amax(max_loss_x) <= (x_max + float(x_max)/100), "the data point {}, max value {}".format(max_loss_x,np.amax(max_loss_x))
        assert np.amin(max_loss_x) >= (x_min - 0.00001), "the data point {}, min value {}".format(max_loss_x,np.amin(max_loss_x))

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
    margins = y_train*(X_train.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    target_clean_total_loss = np.sum(np.maximum(1-margins, 0))

    margins = y_train*(X_train.dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)
    current_clean_total_loss = np.sum(np.maximum(1-margins, 0))

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
    print("----------debug: max loss points------------")

    margins = max_loss_y*(max_loss_x.dot(curr_model.coef_.reshape(-1)) + curr_model.intercept_)
    max_loss_clean = np.maximum(1-margins, 0)
    margins = max_loss_y*(max_loss_x.dot(target_model.coef_.reshape(-1)) + target_model.intercept_)
    max_loss_poison = np.maximum(1-margins, 0)
    print("label of max loss point:",max_loss_y)
    print("loss of max loss point for clean model {}, for poison model {}, their loss difference {}, reg difference {}".format(max_loss_clean,\
        max_loss_poison,max_loss_clean - max_loss_poison,reg_diff))
    print("the exact max loss difference (with regularization):",max_loss_clean - max_loss_poison + args.weight_decay*reg_diff)

    return max_loss_poison
    ########### end of debugging the info ############ 

def incre_online_learning(X_train,
                        y_train,
                        X_test,
                        y_test,
                        curr_model,
                        target_model,
                        x_lim_tuples,
                        args,
                        ScikitModel,
                        target_model_type,
                        attack_num_poison,
                        kkt_tol_par,
                        subpop_data,
                        target_poisons):
    if args.model_type == 'lr':
        print("please set the larning rate and number of optimization steps for logistic regression!")
        if args.dataset == 'dogfish':
            lr = 1e-1
        elif args.dataset == 'adult':
            lr = 0.1
        elif args.dataset == 'mnist_17':
            lr = 0.1
        num_steps = 20000

    X_tar_poison = target_poisons["X_poison"]
    Y_tar_poison = target_poisons["Y_poison"]
    target_num_checker = len(X_tar_poison)

    repeat_num = args.repeat_num # number times we repeat the max loss diff point 
    # info of the subpop
    trn_sub_x,trn_sub_y,trn_nsub_x,trn_nsub_y,\
        tst_sub_x,tst_sub_y,tst_nsub_x,tst_nsub_y = subpop_data

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
    best_loss = -1e10  
    if args.model_type == 'svm':
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
        # compute the conservative lower bound using relaxation for integer programming, for initial model pairs
        if args.dataset == 'adult':
            best_loss_real = -1 
            for cls1 in classes:
                if cls1 == -1:
                    max_loss_real, max_x_real = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[1],args)
                    if best_loss_real < max_loss_real:
                        best_loss_real = max_loss_real
                        max_loss_x_real = max_x_real
                        max_loss_y_real = -1
                else:
                    max_loss_real, max_x_real = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[0],args)
                    if best_loss_real < max_loss_real:
                        best_loss_real = max_loss_real
                        max_loss_x_real = max_x_real
                        max_loss_y_real = 1

    elif args.model_type == 'lr':
        # compute the approximate max loss point
        for cls1 in classes:
            if cls1 == -1:
                max_loss, max_loss_real, max_x = lr_search_max_loss_pt(X_train.shape[1],curr_model,target_model,cls1,x_lim_tuples[1],args,lr=lr,num_steps=num_steps)
                if max_loss_real < max_loss:
                    # this could happen for Adult dataset
                    max_loss_real = max_loss
                if best_loss < max_loss:
                    best_loss = max_loss
                    best_loss_real = max_loss_real
                    max_loss_x = max_x
                    max_loss_y = -1
            else:
                max_loss, max_loss_real, max_x = lr_search_max_loss_pt(X_train.shape[1],curr_model,target_model,cls1,x_lim_tuples[0],args,lr=lr,num_steps=num_steps)
                if max_loss_real < max_loss:
                    # this could happen for Adult dataset
                    max_loss_real = max_loss
                if best_loss < max_loss:
                    best_loss = max_loss
                    best_loss_real = max_loss_real
                    max_loss_x = max_x
                    max_loss_y = 1    

    if args.online_alg_criteria == "max_loss":
        current_tol_par = best_loss
    else:
        # use the euclidean distance as the stop criteria
        current_tol_par = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-theta_ol.reshape(-1))**2+(target_model.intercept_ - bias_ol[0])**2)
    print("tolerance parameter of initial model and target model:",current_tol_par)
    
    tmp_x = np.copy(X_train)
    tmp_y= np.copy(y_train)
    lower_bound = 0
    # set the stop criteria for online learning algorithm
    if target_model_type == "compare":
        ol_lr_threshold = kkt_tol_par
        print("compare the our and kkt attack under same stop croteria!")
    else:
        ol_lr_threshold = args.incre_tol_par
    print("Stop criteria of our poisoning attack:",ol_lr_threshold)
    target_poison_max_losses = []
    max_loss_diffs_reg = []
    ol_tol_params = []
    norm_diffs = []
    current_total_losses = []
    lower_bounds = []
    # append the info on subpop, rest of pop and the whole pop
    trn_sub_acc = []
    trn_nsub_acc = []
    trn_acc = []
    tst_sub_acc = []
    tst_nsub_acc = []
    tst_acc = []

    # record the acc on whole pop and subpop for the first time
    trn_sub_acc1 = curr_model.score(trn_sub_x,trn_sub_y)
    tst_sub_acc1 = curr_model.score(tst_sub_x,tst_sub_y)
    trn_nsub_acc.append(curr_model.score(trn_nsub_x,trn_nsub_y))
    trn_sub_acc.append(trn_sub_acc1)
    trn_acc.append(curr_model.score(X_train,y_train)) 
    tst_nsub_acc.append(curr_model.score(tst_nsub_x,tst_nsub_y))
    tst_sub_acc.append(tst_sub_acc1)
    tst_acc.append(curr_model.score(X_test,y_test)) 

    # print(current_tol_par, ol_lr_threshold, current_tol_par - ol_lr_threshold)
    # assert current_tol_par > ol_lr_threshold

    if args.fixed_budget <= 0:
        if args.require_acc:    
            stop_cond = tst_sub_acc1 > 1-args.err_threshold
        else:
            stop_cond = current_tol_par > ol_lr_threshold
    else:
        stop_cond = num_iter < args.fixed_budget
        print("runing with fixed number of poisoned points,",args.fixed_budget)

    while stop_cond:
    # while trn_sub_acc1 > 1-args.err_threshold:
        print("***** num of poisons and target number of poisons *****:",num_iter,args.fixed_budget)
        print("Current train sub acc:",trn_sub_acc1)
        print("Current test sub acc:",tst_sub_acc1)
        
        print("Ideal Acc on sub:",1-args.err_threshold)
        print("Iteration Number:",num_iter)
        
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
        if args.dataset == 'adult' and args.model_type == 'svm':
            # the relaxed version of the maximum loss diff point
            max_loss_x_real = np.transpose(max_loss_x_real)

        # compute the certified lower bound
        curr_reg = (np.linalg.norm(curr_model.coef_.reshape(-1))**2+(curr_model.intercept_)**2)/2
        target_reg = (np.linalg.norm(target_model.coef_.reshape(-1))**2+(target_model.intercept_)**2)/2
        reg_diff = curr_reg - target_reg
        print("regularizer of current model {}, regularizer of target model {}, reg diff: {}".format(curr_reg,target_reg,reg_diff))

        calib_max_loss_diff = best_loss + args.weight_decay *reg_diff
        total_loss_diff = target_total_loss - current_total_loss - args.weight_decay*X_train.shape[0]*reg_diff
        if args.dataset == 'adult':
            # computed the relaxed maximum loss difference (with regularization)
            calib_max_loss_diff_real = best_loss_real + args.weight_decay *reg_diff

        if total_loss_diff < 0:
            print("Total loss difference is negative, don't update lower bound!")
        elif calib_max_loss_diff < 0:
            print("Best max loss point is negative loss, amazing. don't update lower bound!")
        else:
            print("total train loss difference:",target_total_loss - current_total_loss)
            lower_bound = int((target_total_loss - current_total_loss)/best_loss)
            print("total train loss difference (Reg Included) and max loss diff:",total_loss_diff, calib_max_loss_diff)
            lower_bound = int((total_loss_diff)/(calib_max_loss_diff))
            if args.dataset == 'adult':
                lower_bound_real = int((total_loss_diff)/(calib_max_loss_diff_real))
            else:
                lower_bound_real = lower_bound
            print("the computed lower bound is:",lower_bound)
            print("the conservative lower bound is:",lower_bound_real)

        # record the total loss diffs and max loss diffs
        current_total_losses.append(current_total_loss)
        max_loss_diffs_reg.append(calib_max_loss_diff)
        ol_tol_params.append(current_tol_par)
        # compute the euclidean distance of these models
        target_model_b = target_model.intercept_
        target_model_b = target_model_b[0]
        curr_model_b = curr_model.intercept_
        curr_model_b = curr_model_b[0]
        norm_diff = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-curr_model.coef_.reshape(-1))**2+(target_model_b - curr_model_b)**2)
        norm_diffs.append(norm_diff)

        # append the lower bound w.r.t. iterations 
        lower_bounds.append(lower_bound_real)

        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound
            best_max_loss_x = max_loss_x
            best_max_loss_y = max_loss_y
        if lower_bound_real > conser_lower_bound:
            conser_lower_bound = lower_bound_real

        x_poisons = np.repeat(max_loss_x,repeat_num,axis=0)
        y_poisons = np.array([max_loss_y]*repeat_num)
        
        max_loss_poison = print_for_debug(
                        X_train,
                        y_train,
                        curr_model,
                        target_model,
                        max_loss_x,
                        max_loss_y,
                        x_lim_tuples,
                        args)

        if target_model_type == 'real':
            if lower_bound_real > target_num_checker:
                print("something wrong with the lower bound for target model generated from heuristic method!")
                sys.exit(0)

        if target_model_type in ["kkt","ol"]:
            if lower_bound > attack_num_poison:
                print("something wrong with the lower bound of classifier generated from our attack or KKT attack!")
                sys.exit(0)

        ### update the model weights accordingly
        tmp_x = np.concatenate((tmp_x,x_poisons),axis=0)
        tmp_y = np.concatenate((tmp_y,y_poisons),axis=0)
        print("the updated train set size is:",tmp_x.shape,tmp_y.shape)

        # refit the current model, now the update rule is 
        C = 1.0 / (tmp_x.shape[0] * args.weight_decay)
        # train unpoisoned model
        fit_intercept = True
        curr_model = ScikitModel(
                    C=C,
                    tol=1e-8,
                    fit_intercept=fit_intercept,
                    random_state=args.rand_seed,
                    verbose=False,
                    max_iter = 1000)
        curr_model.fit(tmp_x, tmp_y)
        theta_ol = curr_model.coef_
        bias_ol = curr_model.intercept_
        # update the poisoned dataset
        online_poisons_x.append(x_poisons)
        online_poisons_y.append(y_poisons)

        # record the acc on whole data and subpop
        trn_sub_acc1 = curr_model.score(trn_sub_x,trn_sub_y)
        tst_sub_acc1 = curr_model.score(tst_sub_x,tst_sub_y)

        trn_nsub_acc.append(curr_model.score(trn_nsub_x,trn_nsub_y))
        trn_sub_acc.append(trn_sub_acc1)
        trn_acc.append(curr_model.score(X_train,y_train)) 
        tst_nsub_acc.append(curr_model.score(tst_nsub_x,tst_nsub_y))
        tst_sub_acc.append(tst_sub_acc1)
        tst_acc.append(curr_model.score(X_test,y_test)) 

        # search the max loss difference point with updated model pair
        best_loss = -1 
        if args.model_type == 'svm':
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
                        max_loss_real, max_x_real = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[1],args)
                        if best_loss_real < max_loss_real:
                            best_loss_real = max_loss_real
                            max_loss_x_real = max_x_real
                            max_loss_y_real = -1
                    else:
                        max_loss_real, max_x_real = search_max_loss_pt_contin(curr_model,target_model,cls1,x_lim_tuples[0],args)
                        if best_loss_real < max_loss_real:
                            best_loss_real = max_loss_real
                            max_loss_x_real = max_x_real
                            max_loss_y_real = 1
        elif args.model_type == 'lr':
            # compute the approximate max loss point
            for cls1 in classes:
                if cls1 == -1:
                    max_loss, max_loss_real, max_x = lr_search_max_loss_pt(X_train.shape[1],curr_model,target_model,cls1,x_lim_tuples[1],args,lr=lr,num_steps=num_steps)
                    if max_loss_real < max_loss:
                        # this could happen for Adult dataset
                        max_loss_real = max_loss
                    if best_loss < max_loss:
                        best_loss = max_loss
                        best_loss_real = max_loss_real
                        max_loss_x = max_x
                        max_loss_y = -1
                else:
                    max_loss, max_loss_real, max_x = lr_search_max_loss_pt(X_train.shape[1],curr_model,target_model,cls1,x_lim_tuples[0],args,lr=lr,num_steps=num_steps)
                    if max_loss_real < max_loss:
                        # this could happen for Adult dataset
                        max_loss_real = max_loss
                    if best_loss < max_loss:
                        best_loss = max_loss
                        best_loss_real = max_loss_real
                        max_loss_x = max_x
                        max_loss_y = 1                
    
        if args.online_alg_criteria == "max_loss":
            current_tol_par = best_loss
        else:
            current_tol_par = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-theta_ol.reshape(-1))**2+(target_model.intercept_ - bias_ol[0])**2)
    
        print("current telerance parameter is:",current_tol_par)
        # append the max loss point of max_loss point found w.r.t iterations
        target_poison_max_losses.append(max_loss_poison) 

        # stop condition check
        if args.fixed_budget <= 0:
            if args.require_acc:    
                stop_cond = tst_sub_acc1 > 1-args.err_threshold
            else:
                stop_cond = current_tol_par > ol_lr_threshold  
        else:
            stop_cond = num_iter < args.fixed_budget
    # complete the last iteration info of the our attack
    margins = y_train*(X_train.dot(theta_ol.reshape(-1)) + bias_ol)
    current_total_loss = np.sum(np.maximum(1-margins, 0))
    
    curr_reg = (np.linalg.norm(curr_model.coef_.reshape(-1))**2+(curr_model.intercept_)**2)/2
    target_reg = (np.linalg.norm(target_model.coef_.reshape(-1))**2+(target_model.intercept_)**2)/2
    reg_diff = curr_reg - target_reg
    print("[Final] regularizer of current model {}, regularizer of target model {}, reg diff: {}".format(curr_reg,target_reg,reg_diff))

    calib_max_loss_diff = best_loss + args.weight_decay *reg_diff
    current_total_losses.append(current_total_loss)
    # append the target loss at the final round 
    current_total_losses.append(target_total_loss)
    print("[Final] the pure loss difference of current and target model:",target_total_loss - current_total_loss)
    max_loss_diffs_reg.append(calib_max_loss_diff)
    ol_tol_params.append(current_tol_par)

    # compute the euclidean distance of these models for the final round
    target_model_b = target_model.intercept_
    target_model_b = target_model_b[0]
    curr_model_b = curr_model.intercept_
    curr_model_b = curr_model_b[0]
    norm_diff = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-curr_model.coef_.reshape(-1))**2+(target_model_b - curr_model_b)**2)
    norm_diffs.append(norm_diff)

    target_poison_max_losses = np.array(target_poison_max_losses)
    current_total_losses = np.array(current_total_losses)
    ol_tol_params = np.array(ol_tol_params)
    max_loss_diffs_reg = np.array(max_loss_diffs_reg)
    lower_bounds = np.array(lower_bounds)
    norm_diffs = np.array(norm_diffs)
    online_acc_scores = [trn_acc,trn_sub_acc,trn_nsub_acc,tst_acc,tst_sub_acc,tst_nsub_acc]

    return online_poisons_x, online_poisons_y, best_lower_bound, conser_lower_bound, best_max_loss_x, best_max_loss_y,\
         current_tol_par, np.squeeze(target_poison_max_losses), np.squeeze(current_total_losses), np.squeeze(ol_tol_params),\
              np.squeeze(max_loss_diffs_reg), np.squeeze(lower_bounds), online_acc_scores, norm_diffs

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
    print("conservative lower bound is:",conser_lower_bound)

    print("------performance of KKT attack-------")
    print("maximum loss difference to target classifier:",kkt_tol_par)
    target_model_b = target_model.intercept_
    target_model_b = target_model_b[0]
    kkt_model_p_b = kkt_model_p.intercept_
    kkt_model_p_b = kkt_model_p_b[0]
    kkt_norm_diff = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-kkt_model_p.coef_.reshape(-1))**2+(target_model_b - kkt_model_p_b)**2)
    print("euclidean distance to target classifier:",kkt_norm_diff)
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

    print("------performance of Our attack-------")
    print("maximum loss difference to target classifier:",ol_tol_par)
    target_model_b = target_model.intercept_
    target_model_b = target_model_b[0]
    model_p_online_b = model_p_online.intercept_
    model_p_online_b = model_p_online_b[0]
    ol_norm_diff = np.sqrt(np.linalg.norm(target_model.coef_.reshape(-1)-model_p_online.coef_.reshape(-1))**2+(target_model_b - model_p_online_b)**2)
    print("euclidean distance to target classifier:",ol_norm_diff)
    
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
