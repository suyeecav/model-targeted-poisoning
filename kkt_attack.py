from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import argparse
import time

import numpy as np

import data_utils as data
import datasets
import upper_bounds
import defenses
from upper_bounds import hinge_loss, hinge_grad, logistic_grad

def kkt_setup(
    target_theta,
    target_bias,
    X_train,
    Y_train,
    X_test,
    Y_test,
    dataset_name,
    percentile,
    loss_percentile,
    model,
    model_grad,
    class_map,
    use_slab,
    use_loss,
    use_l2,
    x_pos_tuple = None,
    x_neg_tuple = None,
    model_type='svm'): 

    clean_grad_at_target_theta, clean_bias_grad_at_target_theta = model_grad(
        target_theta,
        target_bias,
        X_train,
        Y_train)
    print(clean_bias_grad_at_target_theta.shape,clean_grad_at_target_theta.shape)
    
    if model_type == 'svm':
        losses_at_target = upper_bounds.indiv_hinge_losses(
            target_theta,
            target_bias,
            X_train,
            Y_train)
    elif model_type == 'lr':
        losses_at_target = upper_bounds.indiv_log_losses(
            target_theta,
            target_bias,
            X_train,
            Y_train)
    else:
        print("please select correct loss")
        raise NameError

    print("ind_log_loss shape",losses_at_target.shape)

    if model_type == 'svm':
        sv_indices = losses_at_target > 0
    else:
        sv_indices = np.arange(X_train.shape[0])

    _, sv_centroids, _, sv_sphere_radii, _ = data.get_data_params(
        X_train[sv_indices, :],
        Y_train[sv_indices],
        percentile=percentile)

    max_losses = [0, 0]
    for y in set(Y_train):
        max_losses[class_map[y]] = np.percentile(losses_at_target[Y_train == y], loss_percentile)

    print('Max losses are: %s' % max_losses)
    model.coef_ = target_theta.reshape((1, -1))
    model.intercept_ = target_bias

    print('If we could get our targeted theta exactly:')
    print('Train            : %.3f' % model.score(X_train, Y_train))
    print('Test (overall)   : %.3f' % model.score(X_test, Y_test))

    if model_type == 'svm':
        two_class_kkt = upper_bounds.TwoClassKKT(
            clean_grad_at_target_theta.shape[0],
            dataset_name=dataset_name,
            X=X_train,
            use_slab=use_slab,
            constrain_max_loss=use_loss,
            use_l2=use_l2,
            x_pos_tuple = x_pos_tuple,
            x_neg_tuple = x_neg_tuple,
            model_type=model_type)
    elif model_type == 'lr':
        # we don't use the cvx solver for logistic regression model
        two_class_kkt = None
    else:
        raise NotImplementedError

    target_bias_grad = clean_bias_grad_at_target_theta
    
    return two_class_kkt, clean_grad_at_target_theta, target_bias_grad, max_losses

def kkt_attack(two_class_kkt,
               target_grad, target_theta,
               total_epsilon, epsilon_pos, epsilon_neg,
               X_train, Y_train,
               class_map, centroids, centroid_vec, sphere_radii, slab_radii,
               target_bias, target_bias_grad, max_losses,
               sv_centroids=None, sv_sphere_radii=None):

    x_pos, x_neg, epsilon_pos, epsilon_neg = two_class_kkt.solve(
        target_grad,
        target_theta,
        epsilon_pos,
        epsilon_neg,
        class_map,
        centroids,
        centroid_vec,
        sphere_radii,
        slab_radii,
        target_bias=target_bias,
        target_bias_grad=target_bias_grad,
        max_losses=max_losses,
        verbose=False)

    obj = np.linalg.norm(target_grad - epsilon_pos * x_pos.reshape(-1) + epsilon_neg * x_neg.reshape(-1))
    print("** Actual objective value: %.4f" % obj)
    num_train = X_train.shape[0]
    total_points_to_add = int(np.round(total_epsilon * X_train.shape[0]))
    num_pos = int(np.round(epsilon_pos * X_train.shape[0]))
    num_neg = total_points_to_add - num_pos
    assert num_neg >= 0

    X_modified, Y_modified = data.add_points(
        x_pos,
        1,
        X_train,
        Y_train,
        num_copies=num_pos)
    X_modified, Y_modified = data.add_points(
        x_neg,
        -1,
        X_modified,
        Y_modified,
        num_copies=num_neg)

    return X_modified, Y_modified, obj, x_pos, x_neg, num_pos, num_neg

def kkt_for_lr(d,args,target_grad,theta_p,bias_p,
    total_eps, eps_pos,eps_neg, X_train, Y_train, x_pos_tuple = None,x_neg_tuple = None,
    lr=1e-5,num_steps=3000,trials=10,optimizer='adam'):
    # we did not implement defenses for KKT for logistic regression
    x_min_pos, x_max_pos = x_pos_tuple
    x_min_neg, x_max_neg = x_neg_tuple

    best_obj = 1e10
    for trial in range(trials):
        # print("------ trial {}------".format(trial))
        # optimization variables 
        if args.dataset == 'dogfish':
            x_pos = np.array([upper_bounds.random_sample(x_min_pos[i],x_max_pos[i]) for i in range(len(x_min_pos))])
            x_neg = np.array([upper_bounds.random_sample(x_min_neg[i],x_max_neg[i]) for i in range(len(x_min_neg))])
        else:
            x_pos = np.array([upper_bounds.random_sample(x_min_pos,x_max_pos) for i in range(d)])
            x_neg = np.array([upper_bounds.random_sample(x_min_neg,x_max_neg) for i in range(d)])

        if optimizer == 'adagrad':
            # store the square of gradients
            grads_squared_pos = np.zeros(d)
            grads_squared_neg = np.zeros(d)
            initial_accumulator_value = 0.001
            grads_squared_pos.fill(initial_accumulator_value)
            grads_squared_neg.fill(initial_accumulator_value)
            epsilon = 1e-7
        elif optimizer == 'adam':
            grads_first_moment_pos = np.zeros(d)
            grads_second_moment_pos = np.zeros(d)
            grads_first_moment_neg = np.zeros(d)
            grads_second_moment_neg = np.zeros(d)

            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
        
        prev_obj = 1e10
        for step in range(num_steps):
            score_pos = np.dot(theta_p, x_pos) + bias_p
            score_neg = np.dot(theta_p, x_neg) + bias_p

            # sigmoid prediction confidence
            prediction_pos = upper_bounds.sigmoid(score_pos) 
            prediction_neg = upper_bounds.sigmoid(score_neg) 
            # output_error_signal_pos = 1 - prediction_pos  # this is also the gradient of b for positive x part
            # output_error_signal_neg = -1 - prediction_neg  # this is also the gradient of b for negative x part

            # the objective value of KKT attack is the norm of following vector
            kkt_obj_grad = target_grad + eps_pos * (1-prediction_pos) * x_pos + eps_neg * (-prediction_neg) * x_neg # note that, we use negative label as 0, not -1
            kkt_obj = np.linalg.norm(kkt_obj_grad)**2
            if step == 0:
                print("(random) initial obj value:",kkt_obj)
            # constant values for x_pos and x_neg
            grad_pos = 2 * eps_pos * (1-prediction_pos) * kkt_obj_grad 
            grad_neg = 2 * eps_neg * (-prediction_neg) * kkt_obj_grad # note that, we use negative label as 0, not -1

            if optimizer == 'gd':
                x_pos -= lr * grad_pos
                x_neg -= lr * grad_neg 
            elif optimizer == 'adagrad':
                """Weights update using adagrad.
                grads2 = grads2 + grads**2
                w' = w - lr * grads / (sqrt(grads2) + epsilon)
                """
                # update x_pos
                grads_squared_pos = grads_squared_pos + grad_pos**2
                x_pos = x_pos - lr * grad_pos / (np.sqrt(grads_squared_pos) + epsilon)
                # update x_neg
                grads_squared_neg = grads_squared_neg + grad_neg**2
                x_neg = x_neg - lr * grad_neg / (np.sqrt(grads_squared_neg) + epsilon)
            elif optimizer == 'adam':
                """Weights update using Adam.
                
                g1 = beta1 * g1 + (1 - beta1) * grads
                g2 = beta2 * g2 + (1 - beta2) * g2
                g1_unbiased = g1 / (1 - beta1**time)
                g2_unbiased = g2 / (1 - beta2**time)
                w = w - lr * g1_unbiased / (sqrt(g2_unbiased) + epsilon)
                """
                time = step + 1
                # update x_pos
                grads_first_moment_pos = beta1 * grads_first_moment_pos + \
                                        (1. - beta1) * grad_pos
                grads_second_moment_pos = beta2 * grads_second_moment_pos + \
                                        (1. - beta2) * grad_pos**2
                grads_first_moment_unbiased_pos = grads_first_moment_pos / (1. - beta1**time)
                grads_second_moment_unbiased_pos = grads_second_moment_pos / (1. - beta2**time)
                x_pos = x_pos - lr * grads_first_moment_unbiased_pos /(np.sqrt(grads_second_moment_unbiased_pos) + epsilon)
                
                # update x_neg
                grads_first_moment_neg = beta1 * grads_first_moment_neg + \
                                        (1. - beta1) * grad_neg
                grads_second_moment_neg = beta2 * grads_second_moment_neg + \
                                        (1. - beta2) * grad_neg**2

                grads_first_moment_unbiased_neg = grads_first_moment_neg / (1. - beta1**time)
                grads_second_moment_unbiased_neg = grads_second_moment_neg / (1. - beta2**time)
                
                x_neg = x_neg - lr * grads_first_moment_unbiased_neg /(np.sqrt(grads_second_moment_unbiased_neg) + epsilon)
            # print(y_tmp,output_error_signal_c, output_error_signal_p)
            # projection step to ensure it is within bounded norm
            x_pos = np.clip(x_pos,x_min_pos,x_max_pos)
            x_neg = np.clip(x_neg,x_min_neg,x_max_neg)
            
            # print("added: min max",np.amin(lr * (gradient_c - gradient_p)),np.amax(lr * (gradient_c - gradient_p)))
            # print("before: min max",np.amin(x),np.amax(x))

            # objective function value found so far (minimization)
            kkt_obj_grad = target_grad + eps_pos * (1-prediction_pos) * x_pos + eps_neg * (-prediction_neg) * x_neg # again, negative label is 0, not -1
            kkt_obj = np.linalg.norm(kkt_obj_grad)**2
            if best_obj > kkt_obj:
                best_obj = kkt_obj
                best_x_pos = x_pos
                best_x_neg = x_neg

            if np.abs(prev_obj - kkt_obj) < 1e-7:
                print("Enough convergence")
                print("steps: {}  current norm (objective): {:.4f}  minimum norm: {:.4f}".format(step+1, kkt_obj, best_obj))
    
                break

            prev_obj = kkt_obj

            # # Print log-likelihood every so often
            # if (step+1) % 2000 == 0:
            #     print("current obj:",kkt_obj)

    print("** Actual objective value: %.4f" % best_obj)
    # num_train = X_train.shape[0]
    total_points_to_add = int(np.round(total_eps * X_train.shape[0]))
    num_pos = int(np.round(eps_pos * X_train.shape[0]))
    num_neg = total_points_to_add - num_pos
    assert num_neg >= 0

    X_modified, Y_modified = data.add_points(
        best_x_pos,
        1,
        X_train,
        Y_train,
        num_copies=num_pos)
    X_modified, Y_modified = data.add_points(
        best_x_neg,
        -1,
        X_modified,
        Y_modified,
        num_copies=num_neg)

    return X_modified, Y_modified, best_obj, best_x_pos, best_x_neg, num_pos, num_neg