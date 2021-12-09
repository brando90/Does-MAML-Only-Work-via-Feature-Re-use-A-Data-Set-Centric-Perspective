#!/homes/miranda9/.conda/envs/automl-meta-learning/bin/python
#PBS -V
#PBS -M brando.science@gmail.com
#PBS -m abe
#PBS -lselect=1:ncpus=112

"""
file for data analysis.
"""
import json

import torch.nn.functional

import matplotlib.pyplot as plt

import torch

import numpy as np
from numpy import random as random

from pathlib import Path

from types import SimpleNamespace

from meta_learning.datasets.dataloader_helper import get_randfnn_dataloader, get_sine_dataloader
from meta_learning.training.meta_training import process_meta_batch

import uutils
# from uutils.torch import set_tracking_running_stats

from tqdm import tqdm

import time

from uutils.torch import compute_result_stats

from pprint import pprint

from argparse import Namespace

from pdb import set_trace as st

def plot(all_diffs_qry, all_diffs_approx_int):
    # plot them both at once
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    # n_bins = expt_sample_size
    n_bins = 30
    axs[0].hist(all_diffs_qry, bins=n_bins)
    axs[0].set_title('functional diff wrt query set')
    axs[0].set_xlabel('functional diff wrt query')
    axs[0].set_ylabel('counts')

    axs[1].hist(all_diffs_approx_int, bins=n_bins)
    axs[1].set_title('functional diff wrt approximate integral')
    axs[1].set_xlabel('functional diff wrt approximate integral')

    plt.show()

def parse_args():
    import argparse
    from uutils import create_logs_dir_and_load, load_cluster_jobids_to
    from datetime import datetime

    parser = argparse.ArgumentParser()

    # experimental setup
    parser.add_argument('--debug', action='store_true', help='if debug')
    # parser.add_argument('--include_synthetic', action='store_true')
    parser.add_argument('--exp_id', type=str)
    # parser.add_argument('--data_path', type=str, default=Path("~/data/lasse_datasets_coq/all-sexpr.feat").expanduser())
    parser.add_argument('--log_root', type=str, default=Path('/logs/').expanduser())

    # for compute stats
    # DONT change this one, has a bug...
    parser.add_argument('--iters', type=int, default=1,  # don't change
                        help='DONT CHANGE keep at 1 unless you fix the iterator issue mentioned bellow.'
                             'It is for the number of times to repeat the data loader looping')
    # CHANGE THIS ONE number of tasks used to compute stats
    parser.add_argument('--iter_tasks', type=int, default=None,  # e.g. 5 out of 199 per I iters (yes change)
                        help='number of tasks actually used from a meta-batch when iterating.')
    parser.add_argument('--meta_batch_size', type=int, default=199,  # e.g. 199 careful changing, it's total # tasks considered
                        help='number of tasks used. Usually 1 minus total # tasks e.g. 199. '
                             'Since we are in eval mode we are using the val dataloader and use this number')
    # cca arg
    # parser.add_argument('--cca_size', type=int, default=None, help='cxa size for any cxa metric (despite name of flag being cca)')

    # Other
    parser.add_argument('--num_workers', type=int, default=8, help='value zero to avoid pycharm errors')

    # parse arguments
    args = parser.parse_args()

    # save number of cpus
    args.cpu_count = torch.multiprocessing.cpu_count()

    # create_logs_dir_and_load(args)
    load_cluster_jobids_to(args)
    # create and load in args path to current experiments from log folder path
    args.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.log_experiment_dir = args.log_root / f'logs_{args.current_time}_jobid_{args.jobid}'
    args.log_experiment_dir.mkdir(parents=True, exist_ok=True)
    # create and load in args path to checkpoint (same as experiment log path)
    args.checkpoint_dir = args.log_experiment_dir

    # set the right meta-batch size for experiments to work (check print statment or all_sims being this size or I*B)
    args.meta_batch_size_train = args.meta_batch_size  # the value of this one doesn't matter for sims eval
    args.meta_batch_size_eval = args.meta_batch_size  # this value matters for sims eval
    return args

def main():
    start = time.time()
    print()
    print('---- running ----')
    print(f'is cuda available: {torch.cuda.is_available()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    args = parse_args()

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
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))

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
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))

    # jobs path for ML + loss vs STD (overfitted sigmoid)
    # path_2_init = Path('~/data/logs/logs_Nov27_15-32-18_jobid_444365.iam-pbs/ckpt_file.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # path_2_init = Path('~/data/logs/logs_Nov27_15-35-48_jobid_444371.iam-pbs/ckpt_file.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # ckpt = torch.load(path_2_init)
    # maml = ckpt['meta_learner']
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))

    # jobs path for ML + loss vs STD (best sigmoid)
    # path_2_init = Path('~/data/logs/logs_Nov27_15-32-18_jobid_444365.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # path_2_init = Path('~/data/logs/logs_Nov27_15-35-48_jobid_444371.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # ckpt = torch.load(path_2_init)
    # maml = ckpt['meta_learner']
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))

    # jobs path for ML + loss vs STD (relu)
    # path_2_init = Path('~/data/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file.pt').expanduser()
    # path_2_init = Path('~/data/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # ckpt = torch.load(path_2_init)
    # maml = ckpt['meta_learner']
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))
    # maml.args.nb_inner_train_steps = 32

    # jobs path for ML + loss vs STD (sigmoid)
    # path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file.pt').expanduser()
    # path_2_init = Path('~/data/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # ckpt = torch.load(path_2_init)
    # maml = ckpt['meta_learner']
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))
    # maml.args.nb_inner_train_steps = 0

    # sinusoid experiment
    # path_2_init = Path('~/data/logs/logs_Nov23_11-26-20_jobid_438708.iam-pbs/ckpt_file.pt').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # ckpt = torch.load(path_2_init)
    # maml = ckpt['meta_learner']
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))
    # # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # tiny std1 for task
    # path_2_init = Path('/Users/brando/data/logs/logs_Dec04_16-16-31_jobid_446021.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # ckpt = torch.load(path_2_init)
    # maml = ckpt['meta_learner']
    # args = Namespace(**vars(args), **vars(ckpt['meta_learner'].args))
    # args.data_path = Path('/Users/brando/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.01_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # # maml.args.inner_lr = 0.01
    # # maml.args.nb_inner_train_steps = 0

    # very small std1=1.0
    path_2_init = Path('~/data/logs/logs_Feb24_12-26-25_jobid_3968/ckpt_file_best_loss.pt').expanduser()
    ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    maml = ckpt['meta_learner']
    args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    args.data_path = Path(
        '/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1e-16_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    args.k_eval = 150
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # very small std1=1e-8
    # path_2_init = Path('~/data/logs/logs_Feb24_12-28-04_jobid_3969/ckpt_file_best_loss.pt').expanduser()
    # ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    # maml = ckpt['meta_learner']
    # args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1e-08_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # very small std1=1e-4
    # path_2_init = Path('~/data/logs/logs_Feb24_13-55-17_jobid_3970/ckpt_file_best_loss.pt').expanduser()
    # ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    # maml = ckpt['meta_learner']
    # args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.0001_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # small std1=1e-2
    # path_2_init = Path('~/data/logs/logs_Dec30_15-48-29_jobid_2387/ckpt_file_best_loss.pt').expanduser()
    # ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    # maml = ckpt['meta_learner']
    # args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.01_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # very small std1=1e-1
    # path_2_init = Path('~/data/logs/logs_Dec24_10-56-46_jobid_451809.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    # maml = ckpt['meta_learner']
    # args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.1_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # args.k_eval = 150
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # very small std1=1.0
    # path_2_init = Path('~/data/logs/logs_Dec27_18-59-02_jobid_451842.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    # maml = ckpt['meta_learner']
    # args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # args.k_eval = 150
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # very small std1=2.0
    path_2_init = Path('~/data/logs/logs_Dec27_18-59-03_jobid_451853.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    maml = ckpt['meta_learner']
    args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    args.data_path = Path(
        '/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    args.k_eval = 150
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # large std1=4.0
    # path_2_init = Path('~/data/logs/logs_Dec30_15-53-29_jobid_2393/ckpt_file_best_loss.pt').expanduser()
    # ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    # maml = ckpt['meta_learner']
    # args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # std1=8.0
    path_2_init = Path('~/data/logs/logs_Feb25_11-32-18_jobid_485168.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    maml = ckpt['meta_learner']
    args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    args.data_path = Path(
        '/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_8.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    args.k_eval = 150
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # std1=16.0
    path_2_init = Path('~/data/logs/logs_Feb25_11-33-01_jobid_485169.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    maml = ckpt['meta_learner']
    args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    args.data_path = Path(
        '/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_16.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    args.k_eval = 150
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # std1=32.0
    path_2_init = Path('~/data/logs/logs_Feb25_11-35-50_jobid_485170.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    maml = ckpt['meta_learner']
    args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    args.data_path = Path(
        '/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_16.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    args.k_eval = 150
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0

    # get adapted init
    args = SimpleNamespace(**{**maml.args.__dict__, **args.__dict__})  # second overwrites the first YES this is what I want
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)
    print(f'-> meta_train_dataloader.batch_size, meta_val_dataloader.batch_size, meta_test_dataloader.batch_size ={meta_train_dataloader.batch_size, meta_val_dataloader.batch_size, meta_test_dataloader.batch_size}')
    # args.k_shots = 10
    # args.k_eval = 15  # used to determine the size of query for approx func difference
    maml.regression()
    if torch.cuda.is_available():
        maml.base_model = maml.base_model.cuda()

    # use the validation data loader not train
    # meta_dataloader = meta_train_dataloader
    meta_dataloader = meta_val_dataloader
    # meta_dataloader = meta_test_dataloader
    print(meta_dataloader.batch_size)

    # layers to do CCA on
    #args.layer_names = ['', '']
    args.layer_names = []
    # for name, _ in maml.base_model.named_parameters():
    for name, m in maml.base_model.named_modules():
        print(f'{name=}')
        if 'fc' in name:
            args.layer_names.append(name)
        # if 'relu' in name or 'fc4_final_l2' in name:
        # if 'relu' in name:
        #         args.layer_names.append(name)
    # args.layer_names.pop(0)
    # args.layer_names.pop()
    print(args.layer_names)

    # compute functional difference
    args.track_higher_grads = False
    print('-- start analysis --')
    print(f'num cpus: {torch.multiprocessing.cpu_count()}')
    print(f'number of workers = {args.num_workers}')
    print(f'--> args.meta_batch_size = {args.meta_batch_size_eval}')
    print(f'--> args.iters = {args.iters}')
    print(f'--> iter_tasks = {args.iter_tasks}')
    print(f'--> args.nb_inner_train_steps = {args.nb_inner_train_steps}')
    print(f'meta_dataloader.batch_size = {meta_dataloader.batch_size}')
    # meta_dataloader.__hash__ = lambda x: random.randrange(1 << 32)

    # looping through the data set multiple times with different examples: https://github.com/tristandeleu/pytorch-meta/issues/112
    e = torch.tensor([])
    all_sims = {'cca': e, 'cka': e, 'nes': e, 'cosine': e, 'nes_output': e, 'query_loss': e}
    with tqdm(range(args.iters)) as pbar:
        it = 0
        while it < args.iters:
            for batch_idx, batch in enumerate(meta_dataloader):
                # batch has a batch of tasks e.g. 200 regression functions or
                print(f'\nit = {it}')
                # print(batch['train'][0].size(0))
                # print(batch['train'][1].size(0))
                spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
                # print(spt_x.mean())
                # print(qry_x.mean())

                # metrics
                # args.iter_tasks = 10
                # sims = maml.functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names, args.iter_tasks)
                sims = maml.functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names)
                print(sims['nes_output'].mean())
                print(sims['nes_output'].std())
                # sims = maml.parallel_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names)
                sims = maml.parallel_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names, iter_tasks=args.iter_tasks)
                print(sims['nes'].size())  # to check [T, L, Keval]
                print(sims['cca'].size())  # to check [T, L]
                print(sims['nes_output'].size())  # to check [T]

                # collect all results to have T=I*B tasks similarities (per layer)
                # note that this code doesn't work quite well without reset the iterator
                # all_sims = {metric: torch.cat([sims[metric], s], dim=0) for metric, s in all_sims.items()}
                for metric, s in all_sims.items():
                    print(metric)
                    all_sims[metric] = torch.cat([sims[metric], s], dim=0)
                print(all_sims['nes'].size())  # to check [T, L, Keval]

                pbar.update()
                it += 1
                if it >= args.iters:
                    break

    # jsut for debugging purposes
    print('\nall_sims = ')
    uutils.pprint_dict(all_sims)
    # compute result statistics
    stats = compute_result_stats(all_sims)

    # save stats json file
    torch.save({'stats': stats, 'all_sims': all_sims}, args.log_experiment_dir / 'stats_and_all_sims.pt')
    with open(args.log_experiment_dir / 'stats.json', 'w') as f:
        json.dump(uutils.to_json(stats), f, indent=4, sort_keys=True)
    with open(args.log_experiment_dir / 'all_sims.json', 'w') as f:
        json.dump(uutils.to_json(all_sims), f, indent=4, sort_keys=True)
    with open(args.log_experiment_dir / 'args.json', 'w') as f:
        json.dump(uutils.to_json(args), f, indent=4, sort_keys=True)

    # print results
    print('-------------------- Results --------------------')
    uutils.pprint_namespace(args)
    print(f'args.nb_inner_train_steps = {args.nb_inner_train_steps}')
    print(f'args.inner_lr = {args.inner_lr}')

    print()
    print(f'--> args.iters = {args.iters} ')
    print(f'--> args.meta_batch_size = {args.meta_batch_size_eval} (this number determines precision of experiments per value)')
    print(f'--> args.iter_tasks: {args.iter_tasks}')
    print(f"--> number of total tasks considered: T = I*B = {all_sims['nes'].size(0)}")
    print(f"--> T, L, keval = {all_sims['nes'].size()}")
    print('--> stats = ')
    uutils.pprint_dict(stats)
    time_passed_msg, _, _, _ = uutils.report_times(start)
    print(time_passed_msg)

if __name__ == '__main__':
    main()
    print('Done!\a')
