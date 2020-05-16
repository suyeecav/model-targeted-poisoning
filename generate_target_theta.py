from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn import linear_model, svm
# from utils import *
from utils import svm_model, calculate_loss, dist_to_boundary
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
weight_decay = 0.09
max_iter = -1

# loss percentile and repeated points, used for indiscriminative attack
# quantile_tape = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
# rep_tape = [1, 2, 5, 10, 20, 30]
quantile_tape = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]
rep_tape = [1, 2, 3, 5, 8, 12, 18, 25]

fit_intercept = True
# def svm_model(**kwargs):
#     return svm.LinearSVC(loss='hinge', **kwargs)

# # hinge loss and error computation
# def calculate_loss(margins):
#     losses = np.amax(1-margins, 0)
#     errs = (margins < 0) + 0.5 * (margins == 0)
#     return np.sum(losses)/len(margins), np.sum(errs)/len(errs)

# # computer distance to decison boundary, used for subpop target generation
# def dist_to_boundary(theta,bias,data):
#     abs_vals = np.abs(np.dot(data,theta) + bias)
#     return abs_vals/(np.linalg.norm(theta,ord = 2))

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
orig_theta = model.coef_.reshape(-1)
orig_bias = model.intercept_
# calculate the clean model acc
train_acc = model.score(X_train,Y_train)
test_acc = model.score(X_test,Y_test)

print(orig_theta.shape,X_train.shape,orig_bias.shape,Y_train.shape)
margins = Y_train*(X_train.dot(orig_theta) + orig_bias)
train_loss, train_err = calculate_loss(margins)
print("train_acc:{}, train loss:{}, train error:{}".format(train_acc,train_loss,train_err))
# test margins and loss
margins = Y_test*(X_test.dot(orig_theta) + orig_bias)
test_loss, test_err = calculate_loss(margins)
print("test_acc:{}, test loss:{}, test error:{}".format(test_acc,test_loss,test_err))

if not subpop:
    ym = (-1)*Y_train
# we prefer points with lower loss (higher loss in correct labels)
margins = Y_train*(X_train.dot(orig_theta) + orig_bias)

y_list = [1,-1]
if not subpop:
    # procedure of generating target classifier, refer to strong poisoning attack paper
    # however, we assume adversaries do not have access to test data
    thetas = []
    biases = []
    train_losses = []
    test_errs = []
    collat_errs = []
    for loss_quantile in quantile_tape:
        for tar_rep in rep_tape:
            print(" ----- Loss Quantile {} and Repetition Number {} ------".format(loss_quantile, tar_rep))
            X_tar = []
            Y_tar = []
            margin_thresh = np.quantile(margins, loss_quantile)
            for i in range(len(y_list)):
                active_cur = np.logical_and(Y_train == y_list[i],margins < margin_thresh)
                X_tar_cur = X_train[active_cur,:]
                y_tar_cur = ym[active_cur]
                # y_orig_cur = Y_test[active_cur]
                X_tar.append(X_tar_cur)
                Y_tar.append(y_tar_cur)
                # Y_orig = Y_orig.append(y_orig_cur)
            X_tar = np.concatenate(X_tar, axis=0)
            Y_tar = np.concatenate(Y_tar, axis=0)
            # repeat points
            X_tar = np.repeat(X_tar, tar_rep, axis=0)
            Y_tar = np.repeat(Y_tar, tar_rep, axis=0) 
            X_train_p = np.concatenate((X_train,X_tar),axis = 0)
            Y_train_p = np.concatenate((Y_train,Y_tar),axis = 0)
            # build another model for poisoned points
            C = 1.0 / (X_train_p.shape[0] * weight_decay)
            model_p = ScikitModel(
                    C=C,
                    tol=1e-10,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 1000)
            model_p.fit(X_train_p,Y_train_p)
            target_theta, target_bias = model_p.coef_.reshape(-1), model_p.intercept_
            # train margin and loss
            margins = Y_train_p*(X_train_p.dot(target_theta) + target_bias)
            train_loss, train_err = calculate_loss(margins)
            train_acc = model_p.score(X_train_p,Y_train_p)
            print("poisoned train acc:{}, train loss:{}, train error:{}".format(train_acc,
            train_loss,train_err))
            margins = Y_train*(X_train.dot(target_theta) + target_bias)
            train_loss, train_err = calculate_loss(margins)
            train_acc = model_p.score(X_train,Y_train)
            print("train acc:{}, train loss:{}, train error:{}".format(train_acc,train_loss,train_err))
            # test margins and loss 
            # # here, we replace test loss with train loss because we cannot use test loss 
            # # to prune the theta, see below
            margins = Y_train*(X_train.dot(target_theta) + target_bias)
            test_loss, test_err = calculate_loss(margins)
            test_acc = model_p.score(X_test,Y_test)
            print("test acc:{}, test loss:{}, test error:{}".format(test_acc,test_loss,test_err))
            # collect the info
            thetas.append(target_theta)
            biases.append(target_bias[0])
            train_losses.append(train_loss)
            test_errs.append(test_err)
            collat_errs.append(test_err)
    thetas = np.array(thetas)
    biases = np.array(biases)
    train_losses = np.array(train_losses)
    test_errs = np.array(test_errs)
    collat_errs = np.array(collat_errs)
    # Prune away target parameters that are not on the Pareto boundary of (train_loss, test_error)
    if prune_theta:
        negtest_errs = [-x for x in test_errs]
        iisort = np.argsort(np.array(negtest_errs))
        iisort_pruned = []
        min_train_loss = 1e9
        ids_remain = []
        for ii in iisort:
            if train_losses[ii] < min_train_loss:
                iisort_pruned.append(ii)
                min_train_loss = train_losses[ii]
        pruned_thetas = thetas[iisort_pruned]
        pruned_biases = biases[iisort_pruned]
        pruned_train_losses = train_losses[iisort_pruned]
        pruned_test_errs = test_errs[iisort_pruned]
        prunned_collat_errs = collat_errs[iisort_pruned]
    # save all params together
    data_all = {}
    data_all['thetas'] = thetas
    data_all['biases'] = biases
    data_all['train_losses'] = train_losses
    data_all['test_errs'] = test_errs
    data_all['collat_errs'] = collat_errs
    # check their shape
    print(test_errs.shape)
    print(collat_errs.shape)
    print(train_losses.shape)
    print(thetas.shape)
    print(biases.shape)
    print(pruned_test_errs.shape)
    print(pruned_train_losses.shape)
    print(pruned_thetas.shape)
    print(pruned_biases.shape)

    data_pruned = {}
    data_pruned['thetas'] = pruned_thetas
    data_pruned['biases'] = pruned_biases
    data_pruned['train_losses'] = pruned_train_losses
    data_pruned['test_errs'] = pruned_test_errs
    data_pruned['collat_errs'] = prunned_collat_errs

    best_theta = thetas[iisort_pruned[0]]
    best_bias = biases[iisort_pruned[0]]
    best_train_loss = train_losses[iisort_pruned[0]]
    best_test_err = test_errs[iisort_pruned[0]]
    best_collat_err = collat_errs[iisort_pruned[0]]
    data_best = {}
    data_best['thetas'] = best_theta
    data_best['biases'] = best_bias
    data_best['train_losses'] = best_train_loss
    data_best['test_errs'] = best_test_err
    data_best['collat_errs'] = best_collat_err
    print("Train Acc of best theta:",1-best_test_err)

    if select_best:
        # choose the one with least train error
        if not os.path.isdir('files/target_classifiers/{}'.format(dataset_name)):
            os.makedirs('files/target_classifiers/{}'.format(dataset_name))
        file_all = open('files/target_classifiers/{}/best_theta_whole'.format(dataset_name), 'wb')
        # dump information to that file
        pickle.dump(data_best, file_all,protocol=2)
        file_all.close()
    else:
        if not os.path.isdir('files/target_classifiers/{}'.format(dataset_name)):
            os.makedirs('files/target_classifiers/{}'.format(dataset_name))
        file_all = open('files/target_classifiers/{}/thetas_whole'.format(dataset_name), 'wb')
        file_pruned = open('files/target_classifiers/{}/thetas_whole_pruned'.format(dataset_name), 'wb')
        # dump information to that file
        pickle.dump(data_all, file_all,protocol=2)
        file_all.close()
        # save pruned thetas
        # dump information to that file
        pickle.dump(data_pruned, file_pruned,protocol=2)
        file_pruned.close()
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
        print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
        print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
        print('Clean Test Target Acc : %.3f' % tst_sub_acc)
        print('Clean Test Collat Acc : %.3f' % model.score(tst_nsub_x, tst_nsub_y))
        # dist to decision boundary
        trn_sub_dist = dist_to_boundary(model.coef_.reshape(-1),model.intercept_,trn_sub_x)
        # try target generated with different ratios
        for kk in range(len(pois_rates)):
            pois_rate = pois_rates[kk]
            x_train_copy, y_train_copy = np.copy(X_train), np.copy(Y_train)
            pois_ct = int(pois_rate * X_train.shape[0])
            print("Poisoned Point:{}, Poisoned Ratio:{},Total Size:{}".format(pois_ct,pois_rate,X_train.shape[0]))
            if pois_ct <= trn_sub_x.shape[0]:
                pois_inds = np.argsort(trn_sub_dist)[:pois_ct]
            else:
                pois_inds = np.random.choice(trn_sub_x.shape[0], pois_ct, replace=True)	
            # generate the poisoning dataset by directly flipping labels
            pois_x, pois_y = trn_sub_x[pois_inds], -trn_sub_y[pois_inds]
            if pois_ct > trn_sub_x.shape[0]:
                y_train_copy = np.delete(y_train_copy,trn_sbcl,axis=0)
                x_train_copy = np.delete(x_train_copy,trn_sbcl,axis=0)
                whole_y = np.concatenate((y_train_copy,pois_y),axis=0)
                whole_x = np.concatenate((x_train_copy,pois_x),axis=0)
            else:
                replace_idx = trn_sbcl[pois_inds]
                y_train_copy[replace_idx] = -y_train_copy[replace_idx]
                whole_x, whole_y = x_train_copy, y_train_copy

            # build another model for poisoned points
            C = 1.0 / (whole_x.shape[0]*weight_decay)
            model_p = ScikitModel(
                    C=C,
                    tol=1e-10,
                    fit_intercept=fit_intercept,
                    random_state=24,
                    verbose=False,
                    max_iter = 1000)
        
            model_p.fit(whole_x,whole_y)
            pois_acc = model_p.score(X_test,Y_test)
            trn_sub_acc = model_p.score(trn_sub_x, trn_sub_y)
            trn_nsub_acc = model_p.score(trn_nsub_x,trn_nsub_y)
            print("Total Acc:",pois_acc)
            print()
            print('Train Target Acc : %.3f' % trn_sub_acc)
            print('Train Collat Acc : %.3f' % trn_nsub_acc)
            print('Test Target Acc : %.3f' % model_p.score(tst_sub_x, tst_sub_y))
            print('Test Collat Acc : %.3f' % model_p.score(tst_nsub_x, tst_nsub_y))
            # theta and bias of the model
            target_theta, target_bias = model_p.coef_.reshape(-1), model_p.intercept_
            thetas.append(target_theta)
            biases.append(target_bias[0])
            if trn_sub_acc == 0:
                if trn_nsub_acc > best_collat_acc:
                    best_collat_acc = trn_nsub_acc
                    best_theta = target_theta
                    best_bias = target_bias[0]
                    print("updated best collat train acc is:",trn_nsub_acc)

        thetas = np.array(thetas)
        biases = np.array(biases)
        data_all = {}
        data_all['thetas'] = thetas
        data_all['biases'] = biases
        data_best = {}
        data_best['thetas'] = best_theta
        data_best['biases'] = best_bias
        print("Acc of best theta and bias:")
        margins = trn_sub_y*(trn_sub_x.dot(best_theta) + best_bias)
        _, train_err = calculate_loss(margins)
        print("Target Train Acc of best theta:",1-train_err)
        margins = trn_nsub_y*(trn_nsub_x.dot(best_theta) + best_bias)
        _, train_err = calculate_loss(margins)
        print("Collat Train Acc of best theta:",1-train_err)        
        # save all the target thetas
        if select_best:
            if not os.path.isdir('files/target_classifiers/{}'.format(dataset_name)):
                os.makedirs('files/target_classifiers/{}'.format(dataset_name))
            file_all = open('files/target_classifiers/{}/best_theta_subpop_{}'.format(dataset_name,int(cl_ind)), 'wb')
            pickle.dump(data_best, file_all,protocol=2)
            file_all.close()
        else:
            if not os.path.isdir('files/target_classifiers/{}'.format(dataset_name)):
                os.makedirs('files/target_classifiers/{}'.format(dataset_name))
            file_all = open('files/target_classifiers/{}/thetas_subpop_{}'.format(dataset_name,int(cl_ind)), 'wb')
            pickle.dump(data_all, file_all,protocol=2)
            file_all.close()