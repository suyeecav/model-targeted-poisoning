import numpy as np
from numpy import genfromtxt
import argparse
import csv
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='svm',help="two models: svm, lr")
parser.add_argument('--dataset', default='adult',help="three datasets: mnist_17, adult, 2d_toy, dogfish")
args = parser.parse_args()

dataset_name = args.dataset
model_type = args.model_type
assert dataset_name in ['adult','mnist_17','2d_toy','dogfish']
assert model_type in ['lr','svm']

if dataset_name == 'mnist_17':
    poison_whole = True
    # see if decreasing by half helps
    incre_tol_par = 0.1
    target_gen_procs = ['orig','improved']
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
elif dataset_name == '2d_toy':
    poison_whole = True
    if poison_whole:
        valid_theta_errs = [0.1,0.15] 
    else:
        valid_theta_errs = [1.0]
elif dataset_name == 'dogfish':
    target_gen_procs = ['orig']
    if model_type == 'lr':
        incre_tol_par = 1.0
    elif model_type == 'svm':
        incre_tol_par = 2.0
    repeat_num = 1 
    poison_whole = True
    rand_seeds = [12,23,34,45]
    if poison_whole:
        valid_theta_errs = [0.1,0.2,0.3]  
    else:
        valid_theta_errs = [0.9]

header_info = ['lower_bound','conserv_lower','kkt_poison_num','online_poison_num','kkt_max_loss_diff','ol_max_loss_diff',\
    'kkt_norm_diff','ol_norm_diff','clean_total_test_acc','clean_subpop_test_acc','clean_collat_test_acc',\
        'clean_total_train_acc','clean_subpop_train_acc','clean_collat_train_acc','target_total_test_acc',\
            'target_subpop_test_acc','target_collat_test_acc','target_total_train_acc','target_subpop_train_acc',\
                'target_collat_train_acc','kkt_total_test_acc','kkt_subpop_test_acc','kkt_collat_test_acc',\
                    'kkt_total_train_acc','kkt_subpop_train_acc','kkt_collat_train_acc','ol_total_test_acc',\
                        'ol_subpop_test_acc','ol_collat_test_acc','ol_total_train_acc','ol_target_train_acc','ol_collat_train_acc']
valid_errs_threshold = []
# now process the files


for target_gen_proc in target_gen_procs:
    for valid_theta_err in valid_theta_errs:
        # write to the average file
        path_name = 'files/final_results/{}/{}/{}/{}'.format(dataset_name,model_type,target_gen_proc,repeat_num)
        if not os.path.isdir(path_name):
            os.makedirs(path_name)
        ave_kkt_lower_bound_file = open('files/final_results/{}/{}/{}/{}/avg_kkt_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,target_gen_proc,repeat_num,incre_tol_par,valid_theta_err), 'w')
        ave_kkt_lower_bound_writer = csv.writer(ave_kkt_lower_bound_file, delimiter=str(' ')) 
        ave_kkt_lower_bound_writer.writerow(header_info)

        ave_real_lower_bound_file = open('files/final_results/{}/{}/{}/{}/avg_real_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,target_gen_proc,repeat_num,incre_tol_par,valid_theta_err), 'w')
        ave_real_lower_bound_writer = csv.writer(ave_real_lower_bound_file, delimiter=str(' ')) 
        ave_real_lower_bound_writer.writerow(header_info+['kkt_norm_grad_diff','ol_norm_grad_diff'])

        ave_ol_lower_bound_file = open('files/final_results/{}/{}/{}/{}/avg_ol_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,target_gen_proc,repeat_num,incre_tol_par,valid_theta_err), 'w')
        ave_ol_lower_bound_writer = csv.writer(ave_ol_lower_bound_file, delimiter=str(' ')) 
        ave_ol_lower_bound_writer.writerow(header_info)
        
        if args.dataset == 'mnist_17' and target_gen_proc == 'orig':
            ave_compare_lower_bound_file = open('files/final_results/{}/{}/{}/{}/avg_compare_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,'orig',repeat_num,incre_tol_par,valid_theta_err), 'w')
            ave_compare_lower_bound_writer = csv.writer(ave_compare_lower_bound_file, delimiter=str(' ')) 
            ave_compare_lower_bound_writer.writerow(header_info)

        kkt_data_all = []
        real_data_all = []
        ol_data_all = []
        compare_data_all = []
        rand_seed_num = len(rand_seeds)
        for rand_seed in rand_seeds:
            print("original data mat shape:")
            # load the files and read average them
            fname = 'files/results/{}/{}/{}/{}/{}/kkt_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,incre_tol_par,valid_theta_err)
            kkt_data = genfromtxt(fname, delimiter=str(' '))
            if dataset_name == 'mnist_17':
                kkt_data = np.expand_dims(kkt_data,axis=0)
            print(kkt_data.shape)
            kkt_data_all.append(kkt_data)
            record_row_num = len(kkt_data)

            fname = 'files/results/{}/{}/{}/{}/{}/real_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,incre_tol_par,valid_theta_err)
            real_data = genfromtxt(fname, delimiter=str(' '))
            if dataset_name == 'mnist_17':
                real_data = np.expand_dims(real_data,axis=0)
            real_data_all.append(real_data)
            print(real_data.shape)

            fname = 'files/results/{}/{}/{}/{}/{}/ol_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,rand_seed,target_gen_proc,repeat_num,incre_tol_par,valid_theta_err)
            ol_data = genfromtxt(fname, delimiter=str(' '))
            if dataset_name == 'mnist_17':
                ol_data = np.expand_dims(ol_data,axis=0)
            ol_data_all.append(ol_data)
            print(ol_data.shape)

            if args.dataset == "mnist_17" and target_gen_proc == 'orig':
                # the compare file is designed to compare KKT with original gen process and our attack with improved target generation process
                fname = 'files/results/{}/{}/{}/{}/{}/compare_lower_bound_and_attacks_tol-{}_err-{}.csv'.format(dataset_name,model_type,rand_seed,'orig',repeat_num,incre_tol_par,valid_theta_err)
                compare_data = genfromtxt(fname, delimiter=str(' '))
                if dataset_name == 'mnist_17':
                    compare_data = np.expand_dims(compare_data,axis=0)
                compare_data_all.append(compare_data)
                print(compare_data.shape)

        # if dataset_name == 'adult':
        kkt_data_all = np.concatenate(kkt_data_all,axis=0)
        real_data_all = np.concatenate(real_data_all,axis=0)
        ol_data_all = np.concatenate(ol_data_all,axis=0)
        if args.dataset == 'mnist_17' and target_gen_proc == 'orig':
            compare_data_all = np.concatenate(compare_data_all,axis=0)
        # elif dataset_name == 'mnist_17':
        #     kkt_data_all = np.array(kkt_data_all)
        #     kkt_data_all = np.squeeze(kkt_data_all)
        #     real_data_all = np.array(real_data_all)
        #     ol_data_all = np.array(ol_data_all)
        print("full data shape:")
        if args.dataset == 'mnist_17' and target_gen_proc == 'orig':
            print(kkt_data_all.shape,real_data_all.shape,ol_data_all.shape,compare_data_all.shape)
        else:
            print(kkt_data_all.shape,real_data_all.shape,ol_data_all.shape)
        
        # compute the mean and std for each of rows
        print("row number of each raw data file:",record_row_num)
       
        for j in range(record_row_num):
            kkt_selected_records = []
            real_selected_records = []
            ol_selected_records = []
            if args.dataset == "mnist_17" and target_gen_proc == 'orig':
                compare_selected_records = []
            for i in range(rand_seed_num):
                kkt_selected_records.append(kkt_data_all[i*record_row_num+j])
                real_selected_records.append(real_data_all[i*record_row_num+j])
                ol_selected_records.append(ol_data_all[i*record_row_num+j])
                if args.dataset == 'mnist_17' and target_gen_proc == 'orig':
                    compare_selected_records.append(compare_data_all[i*record_row_num+j])
            kkt_selected_records = np.array(kkt_selected_records)
            real_selected_records = np.array(real_selected_records)
            ol_selected_records = np.array(ol_selected_records)
            if args.dataset == "mnist_17" and target_gen_proc == 'orig':
                compare_selected_records =  np.array(compare_selected_records)

            print(kkt_selected_records)
            kkt_mean = np.mean(kkt_selected_records,axis=0)
            print(kkt_mean)

            kkt_std = np.std(kkt_selected_records,axis=0)
            ave_kkt_lower_bound_writer.writerow(kkt_mean)
            ave_kkt_lower_bound_writer.writerow(kkt_std)
            
            real_mean = np.mean(real_selected_records,axis=0)
            real_std = np.std(real_selected_records,axis=0)
            ave_real_lower_bound_writer.writerow(real_mean)
            ave_real_lower_bound_writer.writerow(real_std)

            ol_mean = np.mean(ol_selected_records,axis=0)
            ol_std = np.std(ol_selected_records,axis=0)
            ave_ol_lower_bound_writer.writerow(ol_mean)
            ave_ol_lower_bound_writer.writerow(ol_std)

            if args.dataset == 'mnist_17' and target_gen_proc == 'orig':
                compare_mean = np.mean(compare_selected_records,axis=0)
                compare_std = np.std(compare_selected_records,axis=0)
                ave_compare_lower_bound_writer.writerow(compare_mean)
                ave_compare_lower_bound_writer.writerow(compare_std)

        # close the files
        ave_kkt_lower_bound_file.flush()
        ave_kkt_lower_bound_file.close()
        ave_real_lower_bound_file.flush()
        ave_real_lower_bound_file.close()
        ave_ol_lower_bound_file.flush()
        ave_ol_lower_bound_file.close()

        if args.dataset == 'mnist' and target_gen_proc == 'orig':
            ave_compare_lower_bound_file.flush()
            ave_compare_lower_bound_file.close()
        

