import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, 2d_toy")
parser.add_argument('--model_type',default='lr',help='victim model type: SVM or rlogistic regression')
# parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')
# parser.add_argument('--incre_tol_par',default=1e-2,type=float,help='stop value of online alg: max_loss or norm')
args = parser.parse_args()

baseline_only = False # only plot the comparison to the label flipping baseline

# ratio_used_in_target = 0.5
dataset_name = args.dataset
model_type = args.model_type

assert dataset_name in ['adult','mnist_17','2d_toy','dogfish']
assert model_type in ['lr','svm']
# from datasets import load_dataset
# X_train, y_train, X_test, y_test = load_dataset(args.dataset)
# print(X_train.shape)
# print(X_test.shape)
# sys.exit(0)

matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.variant'] = "small-caps"

if dataset_name == 'mnist_17':
    poison_whole = True
    # see if decreasing by half helps
    incre_tol_par = 0.1
    # target_gen_procs = ['orig','improved']
    target_gen_procs = ['orig']
    repeat_num = 1
    valid_theta_errs = [0.05,0.1,0.15]
    # rand_seeds = [12,23,34,45,56]
    rand_seeds = [12]
elif dataset_name == 'adult':
    poison_whole = False
    if args.model_type == 'lr' and not baseline_only:
        incre_tol_par = 0.05
    else:
        incre_tol_par = 0.01
    if baseline_only:
        target_gen_procs = ['improved']
    else:
        target_gen_procs = ['orig']
    repeat_num = 1 
    valid_theta_errs = [1.0]
    # rand_seeds = [12,23,34,45,56]
    rand_seeds = [12]
elif dataset_name == '2d_toy':
    poison_whole = True
    if poison_whole:
        valid_theta_errs = [0.1,0.15] 
    else:
        valid_theta_errs = [1.0]

elif dataset_name == 'dogfish':
    poison_whole = True
    rand_seeds = [12]
    target_gen_procs = ['orig']
    if args.model_type == 'lr':
        incre_tol_par = 1.0
    elif args.model_type == 'svm':
        incre_tol_par = 2.0
    repeat_num = 1 
    if poison_whole:
        valid_theta_errs = [0.1,0.2,0.3]  
    else:
        valid_theta_errs = [0.9]

if poison_whole:
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
    # cls_fname = 'files/data/{}_selected_subpops.txt'.format(dataset_name)
    cls_fname = 'files/data/{}_{}_selected_subpops.txt'.format(dataset_name,args.model_type)
    selected_subpops = np.loadtxt(cls_fname)
    cl_inds = selected_subpops[0]
    cl_cts = selected_subpops[1]


for target_gen_proc in target_gen_procs:
    for rand_seed in rand_seeds:
        for valid_theta_err in valid_theta_errs:
            sub_id = 0
            for kk in range(len(cl_inds)):
                cl_ind = int(cl_inds[kk])
                print("Process ID {} Subpop {} Error {} Rand Seed {} Target Model {}".format(sub_id,cl_ind,valid_theta_err,rand_seed,target_gen_proc))
                # create the valid path
                path_name = 'files/final_results/{}/{}/{}/{}/{}'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num)
                if not os.path.isdir(path_name):
                    os.makedirs(path_name)

                if baseline_only:
                    ratios = [0.2,0.5,1.5]
                    for ratio_used_in_target in ratios:
                        print(" ************* start the plot of ratio {} *************".format(ratio_used_in_target))
                        # if poison_whole:
                        #     filename = 'files/online_models/{}/{}/{}/{}/{}/whole-{}_online_for_baseline_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                        # else:
                        #     filename = 'files/online_models/{}/{}/{}/{}/{}/subpop-{}_online_for_baseline_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                        
                        if poison_whole:
                            filename = 'files/online_models/{}/{}/{}/{}/{}/whole-{}_online_for_baseline_data_tol-{}_err-{}_ratio-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err,ratio_used_in_target)
                        else:
                            filename = 'files/online_models/{}/{}/{}/{}/{}/subpop-{}_online_for_baseline_data_tol-{}_err-{}_ratio-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err,ratio_used_in_target)
                        data_info = np.load(filename)
                        online_acc_scores = data_info["online_acc_scores"]
                        # plot the test acc on the subpopulation, max loss diff, norm diff
                        online_acc_score = online_acc_scores[4]
                        if poison_whole:
                            filename = 'files/kkt_models/{}/{}/{}/{}/{}/whole-{}_baseline_attack_tol-{}_err-{}_ratio-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err,ratio_used_in_target)
                        else:
                            filename = 'files/kkt_models/{}/{}/{}/{}/{}/subpop-{}_baseline_attack_tol-{}_err-{}_ratio-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err,ratio_used_in_target)
                        # if poison_whole:
                        #     filename = 'files/kkt_models/{}/{}/{}/{}/{}/whole-{}_baseline_attack_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                        # else:
                        #     filename = 'files/kkt_models/{}/{}/{}/{}/{}/subpop-{}_baseline_attack_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                        data_info = np.load(filename)
                        baseline_acc_scores = data_info["baseline_acc_scores"]
                        # plot the test acc on the subpopulation, max loss diff, norm diff
                        baseline_acc_score = baseline_acc_scores[4]  
                        print("[Final Acc] Label Flipping: {}, Num of Poisons: {}".format(np.amin(baseline_acc_score),len(baseline_acc_score)))
                        # plot the curve of acc w.r.t. num of poisons
                        matplotlib.rcParams['font.size'] = 28
                        matplotlib.rc('xtick', labelsize=22) 
                        matplotlib.rc('ytick', labelsize=22)
                        plt.xlabel('xlabel', fontsize=28)
                        plt.ylabel('ylabel', fontsize=28)
                        
                        plt.clf()
                        fig = plt.gcf()
                        size = fig.get_size_inches()
                        # print("current size of fig:",size)
                        plt.figure(figsize=(8.8,6.8))

                        if poison_whole:
                            filename = 'files/final_results/{}/{}/{}/{}/{}/baseline_whole-{}_acc_score_tol-{}_err-{}_ratio-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err,ratio_used_in_target)
                        else:
                            filename = 'files/final_results/{}/{}/{}/{}/{}/baseline_subpop-{}_acc_score_tol-{}_err-{}_ratio-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err,ratio_used_in_target)
                        plt.plot(np.arange(len(online_acc_score)), np.squeeze(online_acc_score),'ro', markersize=10)
                        plt.plot(np.arange(len(baseline_acc_score)), np.squeeze(baseline_acc_score),'bs', markersize=10)

                        plt.xlabel('Num of Poisons')
                        if args.dataset in ['mnist_17','dogfish']:
                            plt.ylabel('Test Acc')
                        elif args.dataset == 'adult':
                            plt.ylabel('Test Acc on Subpop')
                        plt.ylim([0,1.01])
                        plt.legend(['Our Attack','Label Flipping on Subpop'])  
                        plt.savefig(filename) 

                    sub_id += 1
                    continue

                # only produce results for the actual target classifier
                if poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                data_info = np.load(filename)
                online_acc_scores = data_info["online_acc_scores"]
                # plot the test acc on the subpopulation, max loss diff, norm diff
                online_acc_score = online_acc_scores[4]
                # max loss diff 
                ol_tol_params = data_info["ol_tol_params"]
                norm_diffs = data_info["norm_diffs"]
                total_loss_on_clean = data_info["current_total_losses"]

                if args.model_type != 'lr':
                    # load the lower bounds of KKT model and our attack model
                    # we do not have the related info because LR cannot find the optimal value of maximum loss difference 
                    if poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/{}/whole-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/{}/subpop-{}_online_for_online_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                    data_info1 = np.load(filename)
                    lower_bounds_ol = data_info1["lower_bounds"]

                    if poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/{}/whole-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/{}/subpop-{}_online_for_kkt_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                    data_info2 = np.load(filename)
                    lower_bounds_kkt = data_info2["lower_bounds"]

                # load the kkt fraction info
                if poison_whole:
                    filename = 'files/kkt_models/{}/{}/{}/{}/{}/whole-{}_kkt_frac_info_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/kkt_models/{}/{}/{}/{}/{}/subpop-{}_kkt_frac_info_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                data_info = np.load(filename)
                kkt_fraction_num_poisons = data_info["kkt_fraction_num_poisons"]
                # key attack info
                kkt_fraction_max_loss_diffs = data_info["kkt_fraction_max_loss_diffs"]
                kkt_fraction_norm_diffs = data_info["kkt_fraction_norm_diffs"]
                kkt_fraction_loss_on_clean = data_info["kkt_fraction_loss_on_clean"]
                kkt_fraction_acc_scores = data_info["kkt_fraction_acc_scores"]
                kkt_fraction_acc_score = kkt_fraction_acc_scores[:,1]
                print(kkt_fraction_acc_scores.shape)
            
                # plot the curve of acc w.r.t. num of poisons
                matplotlib.rcParams['font.size'] = 28
                matplotlib.rc('xtick', labelsize=22) 
                matplotlib.rc('ytick', labelsize=22)
                plt.xlabel('xlabel', fontsize=28)
                plt.ylabel('ylabel', fontsize=28)
                
                plt.clf()
                fig = plt.gcf()
                size = fig.get_size_inches()
                print("current size of fig:",size)
                plt.figure(figsize=(8.8,6.8))

                if poison_whole:
                    filename = 'files/final_results/{}/{}/{}/{}/{}/whole-{}_acc_score_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/final_results/{}/{}/{}/{}/{}/subpop-{}_acc_score_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                plt.plot(np.arange(len(online_acc_score)), np.squeeze(online_acc_score),'ro', markersize=16,label='Our Attack')
                plt.plot(kkt_fraction_num_poisons,kkt_fraction_acc_score,'bs',markersize = 16,label='KKT Attack') 
                # plt.plot(np.arange(len(online_acc_score)), online_acc_score, 'r-')
                # print(kkt_fraction_num_poisons)
                # print(kkt_fraction_acc_score)
                plt.xlabel('Num of Poisons')
                if args.dataset in ['mnist_17','dogfish']:
                    plt.ylabel('Test Acc')
                elif args.dataset == 'adult':
                    plt.ylabel('Test Acc on Subpop')
                plt.ylim([0,1.01])
                # plt.legend(['Our Attack','KKT Attack']) 
                if sub_id == 2 and dataset_name == 'adult' and model_type == 'lr':
                    print("mission complete!") 
                    plt.legend(loc="upper right", bbox_to_anchor=(0.57,0.28))
                    # plt.savefig(filename) 
                    # plt.legend(loc='best')
                else:
                    plt.legend()
                plt.savefig(filename) 
 
                if args.model_type != 'lr': 
                    # logistic regression is heuristic search and hence the lower bound is meaningless
                    matplotlib.rc('xtick', labelsize=16) 
                    matplotlib.rc('ytick', labelsize=16)
                    plt.xlabel('xlabel', fontsize=16)
                    plt.ylabel('ylabel', fontsize=16)
                    plt.clf()
                    fig = plt.gcf()
                    size = fig.get_size_inches()
                    print("current size of fig:",size)
                    plt.figure(figsize=(8.8,6.8))

                    # plot the curve of lower bounds w.r.t. num of poisons
                    if poison_whole:
                        filename = 'files/final_results/{}/{}/{}/{}/{}/whole-{}_lower_bounds_online_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/final_results/{}/{}/{}/{}/{}/subpop-{}_lower_bound_online_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    plt.plot(np.arange(len(lower_bounds_ol)), np.squeeze(lower_bounds_ol), 'r-')
                    # plt.plot(np.arange(len(online_acc_score)), online_acc_score, 'r-')
                    # print(kkt_fraction_num_poisons)
                    # print(kkt_fraction_acc_score)
                    plt.xlabel('Num of Poisons')
                    plt.ylabel('Computed Lower Bound')
                    # plt.legend(['Our Attack','KKT Attack'])  
                    plt.savefig(filename) 

                    if poison_whole:
                        filename = 'files/final_results/{}/{}/{}/{}/{}/whole-{}_lower_bounds_kkt_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/final_results/{}/{}/{}/{}/{}/subpop-{}_lower_bound_kkt_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    plt.plot(np.arange(len(lower_bounds_kkt)), np.squeeze(lower_bounds_kkt), 'r-')
                    # plt.plot(np.arange(len(online_acc_score)), online_acc_score, 'r-')
                    # print(kkt_fraction_num_poisons)
                    # print(kkt_fraction_acc_score)
                    plt.xlabel('Num of Poisons')
                    plt.ylabel('Computed Lower Bound')
                    # plt.legend(['Our Attack','KKT Attack'])  
                    plt.savefig(filename) 

                # plot the curve of max loss diff w.r.t. num of poisons
                matplotlib.rcParams['font.size'] = 34
                matplotlib.rc('xtick', labelsize=30) 
                matplotlib.rc('ytick', labelsize=30)
                plt.xlabel('xlabel', fontsize=30)
                plt.ylabel('ylabel', fontsize=30)

                plt.clf()
                fig = plt.gcf()
                size = fig.get_size_inches()
                print("current size of fig:",size)
                plt.figure(figsize=(14.8,8.7))

                if poison_whole:
                    filename = 'files/final_results/{}/{}/{}/{}/{}/whole-{}_max_loss_diff_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/final_results/{}/{}/{}/{}/{}/subpop-{}_max_loss_diff_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)                
                plt.plot(np.arange(len(ol_tol_params)), np.squeeze(ol_tol_params), 'ro',markersize = 22)
                plt.plot(kkt_fraction_num_poisons,kkt_fraction_max_loss_diffs,'bo',markersize = 22)
                plt.xlabel('Num of Poisons')
                plt.ylabel('Max Loss Diff')
                plt.legend(['Our Attack','KKT Attack']) 
                plt.savefig(filename)   
                
                # plot the curve of norm diff w.r.t num of poisons
                plt.clf()
                fig = plt.gcf()
                size = fig.get_size_inches()
                print("current size of fig:",size)
                plt.figure(figsize=(14.8,8.7))

                if poison_whole:
                    filename = 'files/final_results/{}/{}/{}/{}/{}/whole-{}_norm_diff_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/final_results/{}/{}/{}/{}/{}/subpop-{}_norm_diff_tol-{}_err-{}.png'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)                
                plt.plot(np.arange(len(norm_diffs)), np.squeeze(norm_diffs), 'ro',markersize = 28)
                plt.plot(kkt_fraction_num_poisons,kkt_fraction_norm_diffs,'bo',markersize = 28)
                plt.xlabel('Num of Poisons')
                plt.ylabel('Euclidean Distance')
                plt.legend(['Our Attack','KKT Attack'])  
                plt.savefig(filename)   
                
                # # plot the curve of total loss on clean dataset
                # matplotlib.rc('xtick', labelsize=10) 
                # plt.clf()
                # if poison_whole:
                #     filename = 'files/final_results/{}/{}/{}/{}/whole-{}_total_loss_on_clean_tol-{}_err-{}.png'.format(dataset_name,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)
                # else:
                #     filename = 'files/final_results/{}/{}/{}/{}/subpop-{}_total_loss_on_clean_tol-{}_err-{}.png'.format(dataset_name,rand_seed,target_gen_proc,repeat_num,sub_id,incre_tol_par,valid_theta_err)                
                # plt.plot(np.arange(len(total_loss_on_clean)-1), np.squeeze(total_loss_on_clean[:-1]), 'ro',kkt_fraction_num_poisons+1,kkt_fraction_loss_on_clean,'bo')
                # # print(len(total_loss_on_clean))
                # # print(kkt_fraction_num_poisons)
                # a = kkt_fraction_num_poisons+1
                # assert len(total_loss_on_clean)-1 == a[-1]
                # plt.plot(np.arange(len(total_loss_on_clean)-1),(len(total_loss_on_clean)-1)*[total_loss_on_clean[-1]])
                # plt.xlabel('Num of Poisons')
                # plt.ylabel('Loss on Clean Train Set')
                # plt.legend(['Our Attack','KKT Attack','Target Model'])  
                # plt.savefig(filename)   

                if args.dataset == 'mnist_17' and target_gen_proc == 'orig' and args.model_type == 'svm':
                    matplotlib.rc('xtick', labelsize=22) 
                    matplotlib.rc('ytick', labelsize=22)
                    plt.xlabel('xlabel', fontsize=28)
                    plt.ylabel('ylabel', fontsize=28)
                    matplotlib.rcParams['font.size'] = 28
                
                    # produce results for the comparison of original KKT attack and improved our attack
                    if poison_whole:
                        filename = 'files/online_models/{}/{}/{}/{}/{}/whole-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,'improved',repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/online_models/{}/{}/{}/{}/{}/subpop-{}_online_for_compare_data_tol-{}_err-{}.npz'.format(dataset_name,model_type,rand_seed,'improved',repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                    data_info = np.load(filename)
                    online_acc_scores = data_info["online_acc_scores"]
                    # plot the test acc on the subpopulation, max loss diff, norm diff
                    online_acc_score = online_acc_scores[4]
                    # max loss diff 
                    ol_tol_params = data_info["ol_tol_params"]
                    norm_diffs = data_info["norm_diffs"]
                    total_loss_on_clean = data_info["current_total_losses"]
                
                    # plot the curve of acc w.r.t. num of poisons
                    plt.clf()
                    fig = plt.gcf()
                    size = fig.get_size_inches()
                    print("current size of fig:",size)
                    plt.figure(figsize=(8.8,6.8))

                    if poison_whole:
                        filename = 'files/final_results/{}/{}/{}/{}/{}/whole-{}_acc_score_tol-{}_err-{}_compare.png'.format(dataset_name,model_type,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    else:
                        filename = 'files/final_results/{}/{}/{}/{}/{}/subpop-{}_acc_score_tol-{}_err-{}_compare.png'.format(dataset_name,model_type,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    plt.plot(np.arange(len(online_acc_score)), np.squeeze(online_acc_score), 'ro',markersize = 16)
                    plt.plot(kkt_fraction_num_poisons+1,kkt_fraction_acc_score,'bs',markersize = 16)
                    # plt.plot(np.arange(len(online_acc_score)), online_acc_score, 'r-')
                    # print(kkt_fraction_num_poisons)
                    # print(kkt_fraction_acc_score)
                    plt.xlabel('Num of Poisons')
                    if args.dataset in ['mnist_17','dogfish']:
                        plt.ylabel('Test Acc')
                    elif args.dataset == 'adult':
                        plt.ylabel('Test Acc on Subpop')
                    plt.ylim([0,1.01])
                    plt.legend(['Our Attack','KKT Attack'])  
                    plt.savefig(filename) 

                
                    # # plot the curve of max loss diff w.r.t. num of poisons
                    # plt.clf()
                    # fig = plt.gcf()
                    # size = fig.get_size_inches()
                    # print("current size of fig:",size)
                    # plt.figure(figsize=(14.8,8.7))

                    # if poison_whole:
                    #     filename = 'files/final_results/{}/{}/{}/{}/whole-{}_max_loss_diff_tol-{}_err-{}_compare.png'.format(dataset_name,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    # else:
                    #     filename = 'files/final_results/{}/{}/{}/{}/subpop-{}_max_loss_diff_tol-{}_err-{}_compare.png'.format(dataset_name,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)                
                    # plt.plot(np.arange(len(ol_tol_params)), np.squeeze(ol_tol_params), 'ro',kkt_fraction_num_poisons+1,kkt_fraction_max_loss_diffs,'bo')
                    # plt.xlabel('Num of Poisons')
                    # plt.ylabel('Max Loss Diff')
                    # plt.legend(['Our Attack','KKT Attack'])  
                    # plt.savefig(filename)   
                    
                    # # plot the curve of norm diff w.r.t num of poisons
                    # plt.clf()
                    # fig = plt.gcf()
                    # size = fig.get_size_inches()
                    # print("current size of fig:",size)
                    # plt.figure(figsize=(14.8,8.7))

                    # if poison_whole:
                    #     filename = 'files/final_results/{}/{}/{}/{}/whole-{}_norm_diff_tol-{}_err-{}_compare.png'.format(dataset_name,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    # else:
                    #     filename = 'files/final_results/{}/{}/{}/{}/subpop-{}_norm_diff_tol-{}_err-{}_compare.png'.format(dataset_name,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)                
                    # plt.plot(np.arange(len(norm_diffs)), np.squeeze(norm_diffs), 'ro',kkt_fraction_num_poisons+1,kkt_fraction_norm_diffs,'bo')
                    # plt.xlabel('Num of Poisons')
                    # plt.ylabel('Euclidean Distance')
                    # plt.legend(['Our Attack','KKT Attack'])  
                    # plt.savefig(filename)   
                    
                    # # plot the curve of total loss on clean dataset
                    # matplotlib.rc('xtick', labelsize=10) 
                    # plt.clf()
                    # if poison_whole:
                    #     filename = 'files/final_results/{}/{}/{}/{}/whole-{}_total_loss_on_clean_tol-{}_err-{}_compare.png'.format(dataset_name,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)
                    # else:
                    #     filename = 'files/final_results/{}/{}/{}/{}/subpop-{}_total_loss_on_clean_tol-{}_err-{}_compare.png'.format(dataset_name,rand_seed,'improved',repeat_num,sub_id,incre_tol_par,valid_theta_err)                
                    # plt.plot(np.arange(len(total_loss_on_clean)-1), np.squeeze(total_loss_on_clean[:-1]), 'ro',kkt_fraction_num_poisons+1,kkt_fraction_loss_on_clean,'bo')
                    # # print(len(total_loss_on_clean))
                    # # print(kkt_fraction_num_poisons)
                    # a = kkt_fraction_num_poisons+1
                    # assert len(total_loss_on_clean)-1 == a[-1]
                    # plt.plot(np.arange(len(total_loss_on_clean)-1),(len(total_loss_on_clean)-1)*[total_loss_on_clean[-1]])
                    # plt.xlabel('Num of Poisons')
                    # plt.ylabel('Loss on Clean Train Set')
                    # plt.legend(['Our Attack','KKT Attack','Target Model'])  
                    # plt.savefig(filename)   

                sub_id += 1
                
                

            



