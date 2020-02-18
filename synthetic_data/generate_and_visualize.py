from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import cluster
import csv


def svm_model(**kwargs):
    return svm.LinearSVC(loss='hinge', **kwargs)

generate_dataset = make_classification
class_seps = [1.0,2.0,3.0,4.0,5.0]
weight_decay = 0.09
for class_sep in class_seps:
    print("*************Class Sep: {}****************".format(class_sep))
    # creat files that store clustering info
    info_file = open('files/class_sep{}_info.csv'.format(class_sep), 'w')
    info_writer = csv.writer(info_file, delimiter=str(' ')) 
    
    full_x, full_y = generate_dataset(n_samples = 10000,
                    n_features=2,
                    n_informative=2,
                    n_redundant=0,
                    n_classes=2,
                    n_clusters_per_class=2,
                    flip_y=0.001,
                    class_sep=class_sep,
                    random_state=0)

    print(full_x[:,1].shape)
    print(full_y.shape)

    train_samples = 7000  # Samples used for training the models

    X_train = full_x[:train_samples]
    X_test = full_x[train_samples:]
    y_train = full_y[:train_samples]
    y_test = full_y[train_samples:]
    # convert to {-1,1} as class labels
    y_train = 2*y_train-1
    y_test = 2*y_test-1

    # do clustering and test if these fit previous clusters
    num_clusters = 4
    km = cluster.KMeans(n_clusters=num_clusters,random_state = 0)
    km.fit(X_train)
    trn_km = km.labels_
    tst_km = km.predict(X_test)
    # find the clusters and corresponding subpop size
    cl_inds, cl_cts = np.unique(trn_km, return_counts=True)

    pois_rates = [0.1,0.2,0.3,0.4,0.5]
    ScikitModel = svm_model
    C = 1.0 / (X_train.shape[0] * weight_decay)
    # train unpoisoned model
    model = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=True,
                random_state=24,
                verbose=False)
    model.fit(X_train, y_train)
    clean_acc = model.score(X_test,y_test)
    print("Clean Total Acc:",clean_acc)
    
    # plot the scattred points and decision boundary of clean model
    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1,
                 edgecolor='k')
    plt.xlabel('First Feature')
    plt.ylabel('Second Feature')
    # theta and bias of the model
    theta = model.coef_.reshape(-1)
    bias = model.intercept_
    plot_x_decision = np.array([min(X_test[:,0])-2, max(X_test[:,0])+2])
    plot_y_decision = (-1/theta[1]) * (theta[0] * plot_x_decision + bias)
    plt.plot(plot_x_decision, plot_y_decision, label = "Decision_Boundary")
    # plt.show()
    plt.savefig('figures/classes/class_sep_{}/clean_model.png'.format(class_sep))
    plt.clf()
    # also save the clean model with subpop clusters
    plt.scatter(X_test[:, 0], X_test[:, 1], c=tst_km, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.plot(plot_x_decision, plot_y_decision, label = "Decision_Boundary")
    # plt.show()
    plt.savefig('figures/clusters/class_sep_{}/clean_model.png'.format(class_sep))
    plt.clf()
    
    for cl_ind in cl_inds:
        tst_sbcl = np.where(tst_km==cl_ind)
        trn_sbcl = np.where(trn_km==cl_ind)
        tst_non_sbcl = np.where(tst_km!=cl_ind)
        trn_non_sbcl = np.where(trn_km!=cl_ind)    
        # get the corresponding points in the dataset
        tst_sub_x, tst_sub_y = X_test[tst_sbcl], y_test[tst_sbcl]
        tst_nsub_x, tst_nsub_y = X_test[tst_non_sbcl], y_test[tst_non_sbcl]
        trn_sub_x, trn_sub_y = X_train[trn_sbcl], y_train[trn_sbcl]
        trn_nsub_x, trn_nsub_y = X_train[trn_non_sbcl], y_train[trn_non_sbcl]
        
        # plot the corresponding cluster and remaining clusters, for easiness of visual check
        plt.scatter(tst_sub_x[:, 0], tst_sub_x[:, 1], color = 'red',label = "Subpop {}".format(cl_ind))
        plt.scatter(tst_nsub_x[:, 0], tst_nsub_x[:, 1], color = 'black',label = "Rest of Population")
        plt.xlabel('First Feature')
        plt.ylabel('Second Feature')
        plt.savefig('figures/clusters/class_sep_{}/ref_subpop_{}.png'.format(class_sep,cl_ind))
        plt.clf()
        test_target = model.score(tst_sub_x, tst_sub_y)
        test_collat = model.score(tst_nsub_x, tst_nsub_y)
        print("----------Subpop Indx: {}------".format(cl_ind))
        print('Clean Train Target Acc : %.3f' % model.score(trn_sub_x, trn_sub_y))
        print('Clean Train Collat Acc : %.3f' % model.score(trn_nsub_x,trn_nsub_y))
        print('Clean Test Target Acc : %.3f' % test_target)
        print('Clean Test Collat Acc : %.3f' % test_collat)
        info = [clean_acc,len(trn_sbcl[0]),len(tst_sbcl[0]),test_target,test_collat]
        for pois_rate in pois_rates:
            pois_ct = int(pois_rate * X_train.shape[0])
            print("Poisoned Point:{}, Poisoned Ratio:{},Total Size:{}".format(pois_ct,pois_rate,X_train.shape[0]))
            if pois_ct <= trn_sub_x.shape[0]:
                pois_inds = np.random.choice(trn_sub_x.shape[0], pois_ct, replace=False)
            else:
                pois_inds = np.random.choice(trn_sub_x.shape[0], pois_ct, replace=True)	
            # train on poisoned data
            pois_x, pois_y = trn_sub_x[pois_inds], -trn_sub_y[pois_inds]
            print("Size of Pure Poisoned Data:",pois_x.shape,pois_y.shape)
            whole_x, whole_y = np.concatenate((X_train, pois_x), axis=0), np.concatenate((y_train, pois_y), axis=0)
            print("Shape of full poisoned data:",whole_x.shape[0],whole_y.shape[0])
            C = 1.0 / (whole_x.shape[0] * weight_decay)
            model_p = ScikitModel(
                C=C,
                tol=1e-8,
                fit_intercept=True,
                random_state=24,
                # max_iter=max_iter,
                verbose=False)
            model_p.fit(whole_x, whole_y)
            pois_acc = model_p.score(X_test,y_test)
            print("Total Acc:",pois_acc)
            # acc on subpop and rest of pops
            tst_target_acc = model_p.score(tst_sub_x, tst_sub_y)
            tst_collat_acc = model_p.score(tst_nsub_x, tst_nsub_y)
            print()
            print('Train Target Acc : %.3f' % model_p.score(trn_sub_x, trn_sub_y))
            print('Train Collat Acc : %.3f' % model_p.score(trn_nsub_x,trn_nsub_y))
            print('Test Target Acc : %.3f' % tst_target_acc)
            print('Test Collat Acc : %.3f' % tst_collat_acc)
            info = info + [tst_target_acc,tst_collat_acc]

            # plot the scattred points and decision boundary of poisoned model
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1,
                        edgecolor='k')
            plt.xlabel('First Feature')
            plt.ylabel('Second Feature')
            # theta and bias of the model
            theta = model_p.coef_.reshape(-1)
            bias = model_p.intercept_
            plot_x_decision = np.array([min(X_test[:,0])-1, max(X_test[:,0])+1])
            plot_y_decision = (-1/theta[1]) * (theta[0] * plot_x_decision + bias)
            plt.plot(plot_x_decision, plot_y_decision, label = "Decision_Boundary")
            # plt.show()
            plt.savefig('figures/classes/class_sep_{}/subpop_{}/poison_{}.png'.format(class_sep,cl_ind,pois_rate))
            plt.clf()
            # also save the clean model with subpop clusters
            plt.scatter(X_test[:, 0], X_test[:, 1], c=tst_km, cmap=plt.cm.Set1,
                        edgecolor='k')
            plt.plot(plot_x_decision, plot_y_decision, label = "Decision_Boundary")
            # plt.show()
            plt.savefig('figures/clusters/class_sep_{}/subpop_{}/poison_{}_.png'.format(class_sep,cl_ind,pois_rate))
            plt.clf()

        info_writer.writerow(info)
    info_file.flush()
    info_file.close()


# plt.figure(1)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=trn_km, cmap=plt.cm.Set1,
#             edgecolor='k')
# plt.xlabel('First Feature')
# plt.ylabel('Second Feature')
# plt.show()
# sys.exit(0)


# # Plot the training points
# plt.figure(1)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1,
#             edgecolor='k')
# plt.xlabel('First Feature')
# plt.ylabel('Second Feature')
# plt.show()

# plt.figure(2)
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1,
#             edgecolor='k')
# plt.xlabel('First Feature')
# plt.ylabel('Second Feature')
# plt.show()
