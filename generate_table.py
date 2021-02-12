import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, 2d_toy,dogfish")
parser.add_argument('--model_type',default='lr',help='victim model type: SVM or rlogistic regression')
# parser.add_argument('--poison_whole',action="store_true",help='if true, attack is indiscriminative attack')
# parser.add_argument('--incre_tol_par',default=1e-2,type=float,help='stop value of online alg: max_loss or norm')
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['adult','mnist_17','2d_toy','dogfish']

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
    incre_tol_par = 0.1
    target_gen_procs = ['orig']
    repeat_num = 1
    valid_theta_errs = [0.05,0.1,0.15]
    rand_seeds = [12,23,34,45]
elif dataset_name == 'adult':
    poison_whole = False
    incre_tol_par = 0.01
    target_gen_procs = ['orig']
    repeat_num = 1 
    valid_theta_errs = [1.0]
    rand_seeds = [12,23,34,45]
    if args.model_type == 'lr':
        incre_tol_par = 0.05
elif dataset_name == 'dogfish':
    poison_whole = True
    incre_tol_par = 2.0
    target_gen_procs = ['orig']
    repeat_num = 1 
    valid_theta_errs = [0.1,0.2,0.3]
    rand_seeds = [12,23,34,45]
    if args.model_type == 'lr':
        incre_tol_par = 1.0
elif dataset_name == '2d_toy':
    poison_whole = True
    if poison_whole:
        valid_theta_errs = [0.1,0.15] 
    else:
        valid_theta_errs = [1.0]

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
    cls_fname = 'files/data/{}_selected_subpops.txt'.format(dataset_name)
    selected_subpops = np.loadtxt(cls_fname)
    cl_inds = selected_subpops[0]
    cl_cts = selected_subpops[1]

frac_num = 5
for target_gen_proc in target_gen_procs:
    for valid_theta_err in valid_theta_errs:
        sub_id = 0
        for kk in range(len(cl_inds)):
            cl_ind = int(cl_inds[kk])
            sel_acc_scores = np.zeros((len(rand_seeds),frac_num))
            kkt_acc_scores = np.zeros((len(rand_seeds),frac_num))

            sel_max_loss_diffs = np.zeros((len(rand_seeds),frac_num))
            kkt_max_loss_diffs = np.zeros((len(rand_seeds),frac_num))

            sel_norm_diffs = np.zeros((len(rand_seeds),frac_num))
            kkt_norm_diffs = np.zeros((len(rand_seeds),frac_num))

            num_pts = np.zeros(len(rand_seeds))

            rand_cnt = 0
            for rand_seed in rand_seeds:
                # print("--- collecting info of random seed {} ---".format(rand_seed))
                # print("Process Subpop {} Error {} Rand Seed {} Target Model {}".format(cl_ind,valid_theta_err,rand_seed,target_gen_proc))
                # create the valid path
                path_name = 'files/tables/{}/{}/{}/{}'.format(dataset_name,args.model_type,target_gen_proc,repeat_num)
                if not os.path.isdir(path_name):
                    os.makedirs(path_name)
                
                # only produce results for the actual target classifier
                if poison_whole:
                    filename = 'files/online_models/{}/{}/{}/{}/{}/whole-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/online_models/{}/{}/{}/{}/{}/subpop-{}_online_for_real_data_tol-{}_err-{}.npz'.format(dataset_name,args.model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                data_info = np.load(filename)
                online_acc_scores = data_info["online_acc_scores"]
                # plot the test acc on the subpopulation, max loss diff, norm diff
                online_acc_score = online_acc_scores[4]
                # max loss diff 
                ol_tol_params = data_info["ol_tol_params"]
                norm_diffs = data_info["norm_diffs"]
                total_loss_on_clean = data_info["current_total_losses"]


                # load the kkt fraction info
                if poison_whole:
                    filename = 'files/kkt_models/{}/{}/{}/{}/{}/whole-{}_kkt_frac_info_tol-{}_err-{}.npz'.format(dataset_name,args.model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                else:
                    filename = 'files/kkt_models/{}/{}/{}/{}/{}/subpop-{}_kkt_frac_info_tol-{}_err-{}.npz'.format(dataset_name,args.model_type,rand_seed,target_gen_proc,repeat_num,cl_ind,incre_tol_par,valid_theta_err)
                data_info = np.load(filename)
                kkt_fraction_num_poisons = data_info["kkt_fraction_num_poisons"]
                # key attack info
                kkt_fraction_max_loss_diffs = data_info["kkt_fraction_max_loss_diffs"]
                kkt_fraction_norm_diffs = data_info["kkt_fraction_norm_diffs"]
                kkt_fraction_loss_on_clean = data_info["kkt_fraction_loss_on_clean"]
                kkt_fraction_acc_scores = data_info["kkt_fraction_acc_scores"]
                kkt_fraction_acc_score = kkt_fraction_acc_scores[:,1]
                print(kkt_fraction_acc_scores.shape)
            
                sel_acc_scores[rand_cnt,:] = online_acc_score[kkt_fraction_num_poisons]
                sel_max_loss_diffs[rand_cnt,:] = ol_tol_params[kkt_fraction_num_poisons]
                sel_norm_diffs[rand_cnt,:] = norm_diffs[kkt_fraction_num_poisons]

                num_pts[rand_cnt] = len(online_acc_score)

                kkt_acc_scores[rand_cnt,:] = kkt_fraction_acc_score
                kkt_max_loss_diffs[rand_cnt,:] = kkt_fraction_max_loss_diffs
                kkt_norm_diffs[rand_cnt,:] = kkt_fraction_norm_diffs


                rand_cnt += 1
                sub_id += 1                

            avg_sel_acc_scores = np.mean(sel_acc_scores,0)
            std_sel_acc_scores = np.std(sel_acc_scores,0)
            avg_kkt_acc_scores = np.mean(kkt_acc_scores,0)
            std_kkt_acc_scores = np.std(kkt_acc_scores,0)
            
            avg_sel_max_loss_diffs = np.mean(sel_max_loss_diffs,0)
            std_sel_max_loss_diffs = np.std(sel_max_loss_diffs,0)
            avg_kkt_max_loss_diffs = np.mean(kkt_max_loss_diffs,0)
            std_kkt_max_loss_diffs = np.std(kkt_max_loss_diffs,0)

            avg_sel_norm_diffs = np.mean(sel_norm_diffs,0)
            std_sel_norm_diffs = np.std(sel_norm_diffs,0)
            avg_kkt_norm_diffs = np.mean(kkt_norm_diffs,0)
            std_kkt_norm_diffs = np.mean(kkt_norm_diffs,0)

            ave_num_pt = np.mean(num_pts)

            print("---- for Target Error {} Cluster {} ---- ".format(valid_theta_err,kk))
            print("Number of times averaged: {}".format(len(sel_acc_scores)))
            print("Num of Poisons:",ave_num_pt)
            print("Average Acc Scores:",avg_sel_acc_scores)
            print("Std Acc Scores:",std_sel_acc_scores)
            print("KKT Average Acc Scores:",avg_kkt_acc_scores)
            print("KKT Std Acc Scores:",std_kkt_acc_scores)

            show_more = False
            if show_more:
                print("Average Max Loss Diffs:",avg_sel_max_loss_diffs)
                print("Std Max Loss Diffs:",std_sel_max_loss_diffs)
                print("KKT Average Max Loss Diffs:",avg_kkt_max_loss_diffs)
                print("KKT Std Max Loss Diffs:",std_kkt_max_loss_diffs)



                print("Average Norm Diffs:",avg_sel_norm_diffs)
                print("Std Norm Diffs:",std_sel_norm_diffs)
                print("KKT Average Norm Diffs:",avg_kkt_norm_diffs)
                print("KKT Std Norm Diffs:",std_kkt_norm_diffs)


   

                

                
                

            



