from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn import linear_model, svm
# from utils import *
from utils import svm_model, calculate_loss, dist_to_boundary, cvx_dot
import cvxpy as cvx

import pickle
import argparse
from datasets import load_dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, synthetic")
args = parser.parse_args()

# if true, only select the best target classifier for each subpop
# for subpop attack: it is the one with 0% acc on subpop (train) and minimal damage on rest of pop (train)
# for indiscriminative attack, it is the one with highest train error
select_best = True

# whether poisoning attack is targeted or indiscriminative
if args.dataset == "adult":
    subpop = True
elif args.dataset == "mnist_17":
    subpop = False

if subpop:
    beta_01 = 0
else:
    beta_01 = 0.1

# reduce number of searches on target classifier
prune_theta = True

dataset_name = args.dataset
assert dataset_name in ['adult','mnist_17']

# load data
X_train, Y_train, X_test, Y_test = load_dataset(dataset_name)
if min(Y_test)>-1:
    Y_test = 2*Y_test-1
if min(Y_train) > -1:
    Y_train = 2*Y_train - 1

print(np.amax(Y_train),np.amin(Y_train))
if args.dataset == 'adult':
    weight_decay = 1e-5
elif args.dataset == 'mnist_17':
    weight_decay = 0.09

max_iter = -1
fit_intercept = True

ScikitModel = svm_model
C = 1.0 / (X_train.shape[0] * weight_decay)
model = ScikitModel(
    C=C,
    tol=1e-10,
    fit_intercept=fit_intercept,
    random_state=24,
    verbose=False,
    max_iter = 1000)
model.fit(X_train, Y_train)

model_dumb = ScikitModel(
    C=C,
    tol=1e-10,
    fit_intercept=fit_intercept,
    random_state=24,
    verbose=False,
    max_iter = 1000)
model_dumb.fit(X_train, Y_train)

orig_theta = model.coef_.reshape(-1)
orig_bias = model.intercept_[0]
norm = np.sqrt(np.linalg.norm(orig_theta)**2 + orig_bias**2)
print("norm of clean model:",norm)
# calculate the clean model acc
train_acc = model.score(X_train,Y_train)
test_acc = model.score(X_test,Y_test)

print(orig_theta.shape,X_train.shape,orig_bias.shape,Y_train.shape)
margins = Y_train*(X_train.dot(orig_theta) + orig_bias)
train_loss, train_err = calculate_loss(margins)
reg = (weight_decay/2) * np.linalg.norm(orig_theta)**2
print("train_acc:{}, train loss:{}, train error:{}".format(train_acc,train_loss+reg,train_err))
# test margins and loss
margins = Y_test*(X_test.dot(orig_theta) + orig_bias)
test_loss, test_err = calculate_loss(margins)
print("test_acc:{}, test loss:{}, test error:{}".format(test_acc,test_loss+reg,test_err))

if not subpop:
    ym = (-1)*Y_train
# we prefer points with lower loss (higher loss in correct labels)
margins = Y_train*(X_train.dot(orig_theta) + orig_bias)

class search_target_theta(object):
    def __init__(self,D_c,D_sub):
        # define the bias and variable terms
        X_train, y_train = D_c
        x_sub, y_sub = D_sub
        # n = X_train.shape[0]
        d = X_train.shape[1] # dimension of theta
        # nsub = X_sub.shape[0]
        # print("dimension of data:",d)
        self.cvx_theta_p = cvx.Variable(d)
        self.cvx_bias_p = cvx.Variable() 

        # self.cvx_Dc_X = cvx.Parameter((n,d))
        # self.cvx_Dc_y = cvx.Parameter(n)
        # self.cvx_Dsub_X = cvx.Parameter((nsub,d))
        # self.cvx_Dsub_y = cvx.Parameter(n)
        self.beta_convex = cvx.Parameter()
        # beta = cp.Variable((n,1))
        # v = cp.Variable()
        # loss = cp.sum(cp.pos(1 - cp.mul_elemwise(Y, X @ beta - v)))
        reg = cvx.pnorm(self.cvx_theta_p, 2)**2
        self.cvx_loss = cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(y_train, X_train * self.cvx_theta_p + self.cvx_bias_p)))/X_train.shape[0]\
            + (weight_decay/2) * reg
        # self.cvx_loss = cvx.sum_entries(cvx.maximum(0,1-self.cvx_Dc_y * (cvx_dot(self.cvx_Dc_X,self.cvx_theta_p)) + self.cvx_bias_p))/X_train.shape[0]
        constraint_express = cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(y_sub, x_sub * self.cvx_theta_p + self.cvx_bias_p)))/x_sub.shape[0]
        # constraint_express = cvx.sum(cvx.maximum(0,1-self.cvx_Dsub_y * (cvx_dot(self.cvx_Dsub_X,self.cvx_theta_p)) + self.cvx_bias_p))/X_sub.shape[0]
        self.cvx_constraints = [
            constraint_express <= self.beta_convex 
        ]
        self.cvx_objective = cvx.Minimize(self.cvx_loss)
        self.cvx_prob = cvx.Problem(self.cvx_objective,self.cvx_constraints)

    def solve(self,
            beta_cvx,
            verbose = False
            ):

        # X_train, y_train = D_c
        # x_sub, y_sub = D_sub
        # self.cvx_Dc_X.value = X_train
        # self.cvx_Dc_y = y_train
        # self.cvx_Dsub_X.value = x_sub
        # self.cvx_Dsub_y.value = -y_sub
        self.beta_convex.value = beta_cvx

        self.cvx_prob.solve(verbose=verbose, solver=cvx.GUROBI) 
        Loss_Dc = self.cvx_prob.value
        target_theta = np.array(self.cvx_theta_p.value) 
        target_theta = target_theta.reshape(-1)
        target_bias = np.array(self.cvx_bias_p.value)
        print(target_theta.shape,target_bias,Loss_Dc)
        return target_theta, target_bias, Loss_Dc

y_list = [1,-1]
if not subpop:
    # procedure of generating target classifier, refer to strong poisoning attack paper
    # however, we assume adversaries do not have access to test data

    # Y_train_flip = (-1)*Y_train
    # margins = -Y_train_flip*(X_train.dot(orig_theta) + orig_bias)
    # ave_train_loss, _ = calculate_loss(margins)
    # beta_cvx_low = beta_01
    # beta_cvx_high = ave_train_loss

    valid_thetas = []
    valid_biases = []
    valid_theta_losses = []
    find_target_theta = search_target_theta((X_train,Y_train),
                                                (X_train,-Y_train))

    margins = -Y_train*(X_train.dot(orig_theta) + orig_bias)
    ave_train_loss, _ = calculate_loss(margins)
    margins = Y_train*(X_train.dot(orig_theta) + orig_bias)
    orig_ave_train_loss, _ = calculate_loss(margins)
    print("Average Train loss of clean model:",orig_ave_train_loss)
    beta_cvx_low = beta_01
    beta_cvx_high = np.copy(ave_train_loss)
    bin_search_tol = 1e-2
    print("Intial upper and lower bound of beta_cvx:",beta_cvx_high,beta_cvx_low)
    beta_cvx = (beta_cvx_low+beta_cvx_high)/2
    train_acc = model_dumb.score(X_train,Y_train)
    print("Initial Train Acc :",train_acc)

    print('Clean Train Acc : %.3f' % model.score(X_train, Y_train))
    print('Clean Test Acc : %.3f' % model.score(X_test, Y_test))
    while train_acc > 1 - beta_01 or beta_cvx_high - beta_cvx_low > bin_search_tol:
        print("current beta_cvx is:",beta_cvx)
        target_theta, target_bias, train_loss_reg = find_target_theta.solve(
                                                            beta_cvx,
                                                            verbose = False)
        norm = np.sqrt(np.linalg.norm(target_theta)**2 + target_bias**2)
        print("norm of target theta:",norm)

        model_dumb.coef_ = np.array([target_theta])
        model_dumb.intercept_ = np.array([target_bias])
        train_acc = model_dumb.score(X_train, Y_train)
        print('Target Train Acc : %.3f' % model_dumb.score(X_train, Y_train))
        print('Target Test Acc : %.3f' % model_dumb.score(X_test, Y_test))

        margins = Y_train*(X_train.dot(target_theta) + target_bias)
        train_loss, _ = calculate_loss(margins)
        print("[sanity check] computed reg average train loss:",train_loss + (weight_decay/2) * np.linalg.norm(target_theta)**2)

        if train_acc > 1 - beta_01: # beta_cvx is too high and need to reduce it
            print("beta_cvx is too high, use smaller value to enforce attack goal")
            beta_cvx_high = beta_cvx
        else: #curent beta_cvx satisfies the requirement, can try larger beta_cvx
            print("attack goal achieved, use larger value to enforce theta with lower loss")
            beta_cvx_low = beta_cvx
            valid_thetas.append(target_theta)
            valid_biases.append(target_bias)
            valid_theta_losses.append(train_loss_reg)
        beta_cvx = (beta_cvx_high + beta_cvx_low)/2
        print("beta_cvx upper bound, lower bound, their difference:",beta_cvx_high, beta_cvx_low, beta_cvx_high - beta_cvx_low)

        beta_cvx = (beta_cvx_high + beta_cvx_low)/2
    
    valid_thetas = np.array(valid_thetas)
    valid_biases = np.array(valid_biases)
    valid_theta_losses = np.array(valid_theta_losses)
    print("train losses of valid target thetas")
    print(valid_theta_losses * X_train.shape[0])
    
    data_all = {}
    data_all['valid_thetas'] = valid_thetas
    data_all['valid_biases'] = valid_biases
    data_all['best_theta'] = target_theta
    data_all['best_bias'] = target_bias      
    # save all the target thetas
    if not os.path.isdir('files/target_classifiers/{}'.format(dataset_name)):
        os.makedirs('files/target_classifiers/{}'.format(dataset_name))
    file_all = open('files/target_classifiers/{}/opt_thetas_whole'.format(dataset_name), 'wb')
    # save pruned thetas
    # dump information to that file
    pickle.dump(data_all, file_all,protocol=2)
    file_all.close()

else:
    # do the clustering and attack each subpopulation
    # generation process for subpop: directly flip the labels of subpop
    # choose 5 with highest original acc
    from sklearn import cluster
    num_clusters = 20
    pois_rates = [0.03,0.05,0.1,0.15,0.2,0.3,0.4,0.5]

    cls_fname = 'files/data/{}_trn_cluster_labels.txt'.format(dataset_name)
    if os.path.isfile(cls_fname):
        trn_km = np.loadtxt(cls_fname)
        cls_fname = 'files/data/{}_tst_cluster_labels.txt'.format(dataset_name)
        tst_km = np.loadtxt(cls_fname)
    else:
        num_clusters =20
        km = cluster.KMeans(n_clusters=num_clusters,random_state = 0)
        km.fit(X_train)
        trn_km = km.labels_
        tst_km = km.predict(X_test)
        # save the clustering info to ensure everything is reproducible
        cls_fname = 'files/data/{}_trn_cluster_labels.txt'.format(dataset_name)
        np.savetxt(cls_fname,trn_km)
        cls_fname = 'files/data/{}_tst_cluster_labels.txt'.format(dataset_name)
        np.savetxt(cls_fname,tst_km)
    # find the clusters and corresponding subpop size
    cl_inds, cl_cts = np.unique(trn_km, return_counts=True)
    tst_sub_accs = []
    for i in range(len(cl_cts)):
        cl_ind, cl_ct = cl_inds[i], cl_cts[i]
        print("cluster ID and Size:",cl_ind,cl_ct)         
        # indices of points belong to cluster
        tst_sbcl = np.where(np.logical_and(tst_km==cl_ind,Y_test == -1))
        trn_sbcl = np.where(np.logical_and(trn_km==cl_ind,Y_train == -1))
        tst_non_sbcl = np.where(np.logical_or(tst_km!=cl_ind,Y_test != -1))
        trn_non_sbcl = np.where(np.logical_or(trn_km!=cl_ind,Y_train != -1))
        
        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]  
        tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
        # check the target and collateral damage info
        print("----------Subpop Indx: {} ------".format(cl_ind))
        print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
        print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
        print('Clean Test Target Acc : %.3f' % tst_sub_acc)
        print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))
        tst_sub_accs.append(tst_sub_acc)

    print(cl_inds, cl_cts)
    # print(tst_sub_accs)
    # sort the subpop based on tst acc and choose 5 highest ones
    highest_5_inds = np.argsort(tst_sub_accs)[-5:]
    cl_inds = cl_inds[highest_5_inds]
    cl_cts = cl_cts[highest_5_inds]
    print(cl_inds, cl_cts)

    # save the selected subpop info
    cls_fname = 'files/data/{}_selected_subpops.txt'.format(dataset_name)
    np.savetxt(cls_fname,np.array([cl_inds,cl_cts]))
    print("#---------Selected Subpops------#")
    for i in range(len(cl_cts)): 
        cl_ind, cl_ct = cl_inds[i], cl_cts[i]
        print("cluster ID and Size:",cl_ind,cl_ct)
        thetas = []
        biases = []
        train_losses = []
        test_errs = []
        collat_errs = []         
        best_collat_acc = 0   
        # indices of points belong to cluster
        tst_sbcl = np.where(np.logical_and(tst_km==cl_ind,Y_test == -1))[0]
        trn_sbcl = np.where(np.logical_and(trn_km==cl_ind,Y_train == -1))[0]
        tst_non_sbcl = np.where(np.logical_or(tst_km!=cl_ind,Y_test != -1))[0]
        trn_non_sbcl = np.where(np.logical_or(trn_km!=cl_ind,Y_train != -1))[0]
        
        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], Y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], Y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y  = X_train[trn_sbcl], Y_train[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], Y_train[trn_non_sbcl]  
        tst_sub_acc = model.score(tst_sub_x, tst_sub_y)
        # make sure subpop is from class -1
        assert (tst_sub_y == -1).all()
        assert (trn_sub_y == -1).all()
        # check the target and collateral damage info
        print("----------Subpop Indx: {}------".format(cl_ind))
        print('Clean Total Train Acc : %.3f' % model.score(X_train, Y_train))
        print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
        print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
        print('Clean Total Test Acc : %.3f' % model.score(X_test, Y_test))
        print('Clean Test Target Acc : %.3f' % tst_sub_acc)
        print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))

        # solve the optimization problemand obtain the target classifier,
        # perform binary search on the beta_cvx
        # upper bound for the binary search is set as the avg loss of clean model
        margins = -trn_sub_y*(trn_sub_x.dot(orig_theta) + orig_bias)
        ave_train_loss, _ = calculate_loss(margins)
        margins = trn_sub_y*(trn_sub_x.dot(orig_theta) + orig_bias)
        orig_ave_train_loss, _ = calculate_loss(margins)
        print("Average Train loss of clean model:",orig_ave_train_loss)
        beta_cvx_low = beta_01
        beta_cvx_high = np.copy(ave_train_loss)
        bin_search_tol = 1e-5
        print("Intial upper and lower bound of beta_cvx:",beta_cvx_high,beta_cvx_low)
        beta_cvx = (beta_cvx_low+beta_cvx_high)/2
        trn_sub_acc = model_dumb.score(trn_sub_x, trn_sub_y)
        print("Initial Train Acc on Subpop:",trn_sub_acc)
        valid_thetas = []
        valid_biases = []
        valid_theta_losses = []

        find_target_theta = search_target_theta((X_train,Y_train),(trn_sub_x,-trn_sub_y))
        while trn_sub_acc > beta_01 or beta_cvx_high - beta_cvx_low > bin_search_tol:
            print("current beta_cvx is:",beta_cvx)
            target_theta, target_bias, train_loss_reg = find_target_theta.solve(beta_cvx,
                                                                verbose = False)
            norm = np.sqrt(np.linalg.norm(target_theta)**2 + target_bias**2)
            print("norm of target theta:",norm)

            model_dumb.coef_ = np.array([target_theta])
            model_dumb.intercept_ = np.array([target_bias])
            # print out the acc info on subpop
            trn_sub_acc = model_dumb.score(trn_sub_x, trn_sub_y)
            # print("----------Subpop Indx: {}------".format(cl_ind))
            print('Target Total Train Acc : %.3f' % model_dumb.score(X_train, Y_train))
            print('Target Train Target Acc : %.3f' % model_dumb.score(trn_sub_x, trn_sub_y))
            print('Target Train Collat Acc : %.3f' % model_dumb.score(trn_nsub_x,trn_nsub_y))
            print('Target Total Test Acc : %.3f' % model_dumb.score(X_test, Y_test))
            print('Target Test Target Acc : %.3f' % model_dumb.score(tst_sub_x, tst_sub_y))
            print('Target Test Collat Acc : %.3f' % model_dumb.score(tst_nsub_x, tst_nsub_y))

            margins = Y_train*(X_train.dot(target_theta) + target_bias)
            train_loss, _ = calculate_loss(margins)
            print("[sanity check] computed reg average train loss:",train_loss + (weight_decay/2) * np.linalg.norm(target_theta)**2)
            
            # margins = Y_test*(X_test.dot(target_theta) + target_bias)
            # test_loss, _ = calculate_loss(margins) + weight_decay * np.linalg.norm(target_theta)**2
            # print("Average train and test loss of Valid Target Theta (with reg):",train_loss, test_loss)

            # margins = Y_train*(X_train.dot(model.coef_.reshape(-1)) + model.intercept_)
            # train_loss, _ = calculate_loss(margins)
            # margins = Y_test*(X_test.dot(model.coef_.reshape(-1)) + model.intercept_)
            # test_loss, _ = calculate_loss(margins)
            # print("Average train and test loss of Valid clean model:",train_loss, test_loss)

            if trn_sub_acc > beta_01: # beta_cvx is too high and need to reduce it
                print("beta_cvx is too high, use smaller value to enforce attack goal")
                beta_cvx_high = beta_cvx
            else: #curent beta_cvx satisfies the requirement, can try larger beta_cvx
                print("attack goal achieved, use larger value to enforce theta with lower loss")
                beta_cvx_low = beta_cvx
                valid_thetas.append(target_theta)
                valid_biases.append(target_bias)
                valid_theta_losses.append(train_loss_reg)
            beta_cvx = (beta_cvx_high + beta_cvx_low)/2
            print("beta_cvx upper bound, lower bound, their difference:",beta_cvx_high, beta_cvx_low, beta_cvx_high - beta_cvx_low)

        valid_thetas = np.array(valid_thetas)
        valid_biases = np.array(valid_biases)
        valid_theta_losses = np.array(valid_theta_losses)
        print("train losses of valid target thetas")
        print(valid_theta_losses * X_train.shape[0])
        data_all = {}
        data_all['valid_thetas'] = valid_thetas
        data_all['valid_biases'] = valid_biases
        data_all['best_theta'] = target_theta
        data_all['best_bias'] = target_bias      
        # save all the target thetas
        if not os.path.isdir('files/target_classifiers/{}'.format(dataset_name)):
            os.makedirs('files/target_classifiers/{}'.format(dataset_name))
        file_all = open('files/target_classifiers/{}/opt_thetas_subpop_{}'.format(dataset_name,int(cl_ind)), 'wb')
        pickle.dump(data_all, file_all,protocol=2)
        file_all.close()


