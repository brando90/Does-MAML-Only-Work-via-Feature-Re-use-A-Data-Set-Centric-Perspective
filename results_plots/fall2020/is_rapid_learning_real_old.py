# %%

import torch.nn.functional

import matplotlib.pyplot as plt

# integral difference between || f_init - Adapt(f_init) ||^2

import torch

import numpy as np

from pathlib import Path

from types import SimpleNamespace

from meta_learning.datasets.dataloader_helper import get_randfnn_dataloader, get_sine_dataloader
from meta_learning.training.meta_training import process_meta_batch

import uutils
# from uutils.torch import set_tracking_running_stats

# from uutils.torch import cca

from tqdm import tqdm

import time

from pdb import set_trace as st

def print_results(args, all_meta_eval_losses, all_diffs_cka, diffs_cka_all, all_diffs_neds, all_diffs_qry, all_diffs_r2_avg):
    print(f'Meta Val loss (using query set of course, (k_val = {args.k_eval}))')
    meta_val_loss_mean = np.average(all_meta_eval_losses)
    meta_val_loss_std = np.std(all_meta_eval_losses)
    print(f'-> meta_val_loss = {meta_val_loss_mean} +-{meta_val_loss_std}')

    print(f'\nFuntional difference according to query set, (approx integral with k_val = {args.k_eval})')
    diff_qry_mean = np.average(all_diffs_qry)
    diff_qry_std = np.std(all_diffs_qry)
    print(f'-> diff_qrt_mean = {diff_qry_mean} +-{diff_qry_std}')

    print(f'Funtional difference according to cca (k_val = {args.k_eval})')
    diff_cca_mean = np.average(all_diffs_cca)
    diff_cca_std = np.std(all_diffs_cca)
    print(f'-> diff_cca_mean = {diff_cca_mean} +-{diff_cca_std}')

    print(f'Funtional difference according to cka (k_val = {args.k_eval})')
    diff_cka_mean = np.average(all_diffs_cka)
    diff_cka_std = np.std(all_diffs_cka)
    print(f'-> diff_cka_mean = {diff_cka_mean} +-{diff_cka_std}')

    # print(f'Funtional difference according to cka (k_val = {args.k_eval})')
    # diff_cka_mean = np.average(all_diffs_cka)
    # diff_cka_std = np.std(all_diffs_cka)
    # print(f'-> diff_cca_mean = {diff_cka_mean} +-{diff_cka_std}')

    print(f'Funtional difference according to ned (k_val = {args.k_eval})')
    diff_ned_mean = np.average(all_diffs_neds)
    diff_ned_std = np.std(all_diffs_neds)
    print(f'-> diff_ned_mean = {diff_ned_mean} +-{diff_ned_std}')

    # print(f'Funtional difference according to r2s_avg (k_val = {args.k_eval})')
    # diff_r2avg_mean = np.average(all_diffs_r2_avg)
    # diff_r2avg_std = np.std(all_diffs_r2_avg)
    # print(f'-> diff_r2avg_mean = {diff_r2avg_mean} +-{diff_r2avg_std}')

    # print(f'Funtional difference according to integral approx')
    # diff_approx_int_mean = np.average(all_diffs_approx_int)
    # diff_approx_int_std = np.std(all_diffs_approx_int)
    # print(f'-> diff_qrt_mean = {diff_approx_int_mean} +-{diff_approx_int_std}')

    # print(f'Funtional difference according to r2_1_mse_var')
    # diff_r2_1_mse_var_mean = np.average(all_diffs_r2_1_mse_var)
    # diff_r2_1_mse_var_std = np.std(all_diffs_r2_1_mse_var)
    # print(f'-> diff_ned_mean = {diff_r2_1_mse_var_mean} +-{diff_r2_1_mse_var_std}')

start = time.time()

print()
print('---- running ----')
print(f'is cuda available: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

args = SimpleNamespace()

# debug init
# logs/logs_Nov27_15-31-36_jobid_444362.iam-pbs/ckpt_file.pt
# path_2_init = Path('~/data/logs/logs_Nov17_13-57-11_jobid_416472.iam-pbs/ckpt_file.pt').expanduser()  # debugging ckpt

# jobs path for ML + loss vs STD (overfitted)
# path_2_init = Path('~/data/logs/logs_Nov27_15-31-36_jobid_444362.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov30_03-38-25_jobid_444368.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-36-34_jobid_444374.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-39-49_jobid_444380.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# # jobs path for ML + loss vs STD (best)
# path_2_init = Path('~/data/logs/logs_Nov27_15-31-36_jobid_444362.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov30_03-38-25_jobid_444368.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-36-34_jobid_444374.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-39-49_jobid_444380.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (overfitted sigmoid)
# path_2_init = Path('~/data/logs/logs_Nov27_15-32-18_jobid_444365.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-35-48_jobid_444371.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (best sigmoid)
# path_2_init = Path('~/data/logs/logs_Nov27_15-32-18_jobid_444365.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-35-48_jobid_444371.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (relu)
# path_2_init = Path('~/data/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file.pt').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# maml.args.nb_inner_train_steps = 32
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (sigmoid)
# path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file.pt').expanduser()
# path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# maml.args.nb_inner_train_steps = 0
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# sinusoid experiment
# path_2_init = Path('~/data/logs/logs_Nov23_11-26-20_jobid_438708.iam-pbs/ckpt_file.pt').expanduser()
# args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
# args.num_workers = 0
# # maml.args.inner_lr = 0.01
# maml.args.nb_inner_train_steps = 0
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_sine_dataloader(args)

# tiny std1 for task
# path_2_init = Path('/Users/brando/data/logs/logs_Dec04_16-16-31_jobid_446021.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# ckpt = torch.load(path_2_init)
# maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
# args.data_path = Path('/Users/brando/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.01_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# args.num_workers = 0
# # maml.args.inner_lr = 0.01
# # maml.args.nb_inner_train_steps = 0
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# tiny std1 for task
# path_2_init = Path('~/data/logs/logs_Dec30_15-48-29_jobid_2387/ckpt_file_best_loss.pt').expanduser()
# ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
# maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
# args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.01_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
# args.num_workers = 0
# # maml.args.inner_lr = 0.01
# # maml.args.nb_inner_train_steps = 0
# args.meta_batch_size_train = 200; args.meta_batch_size_eval = args.meta_batch_size_train
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)
# print(meta_val_dataloader.batch_size)

# large std1 for task
path_2_init = Path('/logs/logs_Dec30_15-53-29_jobid_2393/ckpt_file_best_loss.pt').expanduser()
ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
args.data_path = Path(
    '/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
args.num_workers = 0
# maml.args.inner_lr = 0.01
# maml.args.nb_inner_train_steps = 0
args.meta_batch_size_train = 200; args.meta_batch_size_eval = args.meta_batch_size_train
meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)
print(meta_val_dataloader.batch_size)

# get adapted init
args = SimpleNamespace(**{**maml.args.__dict__, **args.__dict__})  # second overwrites the first YES this is what I want
# args.k_shots = 10
# args.k_eval = 15  # used to determine the size of query for approx func difference
maml.regression()
if torch.cuda.is_available():
    maml.base_model = maml.base_model.cuda()

# meta_dataloader = meta_train_dataloader
meta_dataloader = meta_val_dataloader
# meta_dataloader = meta_test_dataloader
print(meta_dataloader.batch_size)

# layers to do CCA on
#args.layer_names = ['', '']
args.layer_names = []
# for name, _ in maml.base_model.named_parameters():
for name, m in maml.base_model.named_modules():
    if 'fc' in name:
        args.layer_names.append(name)
# args.layer_names.pop(0)
# args.layer_names.pop()
print(args.layer_names)

# compute functional difference
args.track_higher_grads = False
args.iters = 10
# metrics
all_meta_eval_losses = []
all_diffs_qry = []
all_diffs_cca = []
all_diffs_cka = []
all_diffs_neds = []

all_diffs_approx_int = []
all_diffs_r2_1_mse_var = []
all_diffs_r2_avg = []

print('-- start analysis --')
print(f'number of workers = {args.num_workers}')
print(f'--> args.meta_batch_size = {args.meta_batch_size_eval}')
print(f'--> args.iters = {args.iters}')
print(f'--> args.nb_inner_train_steps = {args.nb_inner_train_steps}')

print(meta_dataloader.batch_size)
with tqdm(range(args.iters)) as pbar:
    it = 0
    while it < args.iters:
        for batch_idx, batch in enumerate(meta_dataloader):
            print(f'it = {it}')
            # print(batch['train'][0].size(0))
            # print(batch['train'][1].size(0))
            # print(meta_dataloader.batch_sizeresults_plots/is_rapid_learning_real.py)
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

            meta_eval_loss, meta_eval_acc = maml(spt_x, spt_y, qry_x, qry_y)
            diffs_qry = maml.functional_difference(spt_x, spt_y, qry_x, qry_y, type_diff='use_qry')
            # diffs_cca_all, cca_average_per_layer = maml.functional_difference(spt_x, spt_y, qry_x, qry_y,
            #                                                               type_diff='pwcca', layer_names=args.layer_names,
            #                                                               iter_tasks=4)
            # diffs_cka_all, cka_average_per_layer = maml.functional_difference(spt_x, spt_y, qry_x, qry_y,
            #                                                               type_diff='lincka', layer_names=args.layer_names,
            #                                                               iter_tasks=4)
            diffs_ned = maml.functional_difference(spt_x, spt_y, qry_x, qry_y, type_diff='ned')

            # diffs_cca_all, average_per_layer = maml.functional_difference(spt_x, spt_y, qry_x, qry_y,
            #                                                               type_diff='svcca', layer_names=args.layer_names,
            #                                                               iter_tasks=4)
            # diffs_approx_int = maml.functional_difference(spt_x, spt_y, qry_x, qry_y, type_diff='approx_int')
            # diffs_r2_1_mse_var = maml.functional_difference(spt_x, spt_y, qry_x, qry_y, type_diff='1_minus_total_residuals')
            # diffs_r2_avgs = maml.functional_difference(spt_x, spt_y, qry_x, qry_y, type_diff='average_r2s')
            # diffs_ned = maml.functional_difference(spt_x, spt_y, qry_x, qry_y, type_diff='mohalanobis')

            # use avg of tasks
            all_meta_eval_losses.append(meta_eval_loss)
            all_diffs_qry.append(diffs_qry)

            # all_diffs_cca.append(diffs_cca_all)
            # all_diffs_cka.append(diffs_cka_all)
            all_diffs_neds.append(diffs_ned)

            # all_diffs_approx_int.append(diffs_approx_int)
            # all_diffs_r2_1_mse_var.append(diffs_r2_1_mse_var)
            # all_diffs_r2_avg.append(diffs_r2_avgs)

            # not using avg of tasks
            # all_diffs_qry.extend(diffs_qry)
            # all_diffs_approx_int.extend(diffs_approx_int)

            # print_results(args, all_meta_eval_losses, all_diffs_cca, all_diffs_cka, all_diffs_neds, all_diffs_qry, all_diffs_cka)

            it += 1
            pbar.update()
            if it >= args.iters:
                break

print('-------------------- Results --------------------')
print(f'{len(all_diffs_neds)=}')
print(f'\nargs = {args}')
print(f'\nargs.meta_batch_size = {args.meta_batch_size_eval} (this number determines precision of experiments per value)')
print(f'args.iters = {args.iters} (total number of expt values)')
print(f'--> args.nb_inner_train_steps = {args.nb_inner_train_steps}')
print(f'--> args.inner_lr = {args.inner_lr}')

print_results(args, all_meta_eval_losses, all_diffs_cca, all_diffs_cka, all_diffs_neds, all_diffs_qry, all_diffs_cka)

# check some values
# print()
# print('-- diffs values --')
# print(f'all_diffs_qry = {all_diffs_qry}')
# print(f'all_diffs_cca = {all_diffs_cca}')
# print(f'all_diffs_neds = {all_diffs_neds}')
# print(f'all_diffs_r2_1_mse_var = {all_diffs_r2_1_mse_var}')

# # plot them both at once
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#
# # We can set the number of bins with the `bins` kwarg
# # n_bins = expt_sample_size
# n_bins = 30
# axs[0].hist(all_diffs_qry, bins=n_bins)
# axs[0].set_title('functional diff wrt query set')
# axs[0].set_xlabel('functional diff wrt query')
# axs[0].set_ylabel('counts')
#
# axs[1].hist(all_diffs_approx_int, bins=n_bins)
# axs[1].set_title('functional diff wrt approximate integral')
# axs[1].set_xlabel('functional diff wrt approximate integral')
#
# plt.show()


# REPORT TIME

time_passed_msg, _, _, _ = uutils.report_times(start)
print(time_passed_msg)