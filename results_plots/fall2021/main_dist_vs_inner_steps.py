#%%
"""
Main script for computing dist(f, A(f)) vs inner steps for increasing inner steps for A=MAML.
"""
from argparse import Namespace

import progressbar
import torch

from pathlib import Path

from types import SimpleNamespace

from meta_learning.datasets.dataloader_helper import get_randfnn_dataloader, get_sine_dataloader
from meta_learning.training.meta_training import process_meta_batch, log_sim_to_check_presence_of_feature_reuse

import uutils

from tqdm import tqdm

import time

from pprint import pprint

from pdb import set_trace as st

from uutils import torch_uu
from uutils.torch_uu.distributed import is_lead_worker

start = time.time()

def get_args_for_experiment() -> Namespace:
    # - get my default args
    args = uutils.parse_basic_meta_learning_args_from_terminal()
    args.log_to_wandb = False
    # args.log_to_wandb = True
    args.wandb_project = 'meta-learning-neurips-workshop'
    # args.experiment_name = 'debug_test'
    # args.experiment_name = 'dist_vs_std_playground'
    # args.experiment_name = 'dist_vs_std-relu-best'
    # args.experiment_name = 'dist_vs_std-relu-overfitted'
    # args.experiment_name = 'dist_vs_std-relu-sinusoid'
    # args.experiment_name = 'dist_vs_inner-steps-simoid-best-std1=2p0'
    args.experiment_name = 'dist_vs_inner-steps-relu-best-std1=2p0'
    args.run_name = 'inner-steps=0'
    args.run_name = 'inner-steps=1'
    args.run_name = 'inner-steps=2'
    args.run_name = 'inner-steps=4'
    args.run_name = 'inner-steps=8'
    args.run_name = 'inner-steps=16-2nd-try'
    # args.run_name = 'inner-steps=32'
    args = uutils.setup_args_for_experiment(args)

    # - my args
    args.num_workers = 0

    args.metrics_as_dist = True
    args.num_its = 1
    args.show_layerwise_sims = False  # only show representation sims
    args.meta_batch_size_train = 30; args.meta_batch_size_eval = args.meta_batch_size_train
    args.k_eval = 100  # needed to avoid cca to throw errors

    # -- job for sinusoid expt (didn't do much with it actually...)
    # args.path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov23_11-26-20_jobid_438708.iam-pbs/ckpt_file.pt').expanduser()
    # args.data_path = 'sinusoid'
    # args.to_single_float_float32 = True  # not sure if generally needed but data seems to be double while ckpt was float32...weird, fine lets make it work
    # args.training_mode = 'iterations'

    # -- jobs path for ML + loss vs STD (sigmoid)
    # ###path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file.pt').expanduser()
    args.path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    args.data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()

    # -- jobs path for ML + loss vs STD (relu)
    # ###path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file.pt').expanduser()
    args.path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file_best_loss.pt').expanduser()
    # args.data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()

    # -- print path to init & path to data
    print(f'{args.path_2_init=}')
    print(f'{args.data_path=}')

    # -- INNER STEPS NEED TO BE EDITED IN EDIT_HERE LOCATION
    return args

def get_meta_learner(args: Namespace):
    ckpt = torch.load(args.path_2_init)
    meta_learner = ckpt['meta_learner']
    meta_learner.regression()
    if torch.cuda.is_available():
        meta_learner.base_model = args.meta_learner.base_model.cuda()
    return meta_learner

def main_run_expt():
    # - get args & merge them with the args of actual experiment run
    args: Namespace = get_args_for_experiment()
    print(f'{args.data_path=}')
    args.meta_learner = get_meta_learner(args)
    args = uutils.merge_args(starting_args=args.meta_learner.args, updater_args=args)
    # args = uutils.merge_args(starting_args=args, updater_args=args.meta_learner.args)
    args.meta_learner.args = args  # to avoid meta learner running with args only from past experiment and not with metric analysis experiment
    pprint(f'{args.meta_learner.args=}')

    # -- EDIT_HERE
    args.meta_learner.args.nb_inner_train_steps = 0
    args.meta_learner.args.nb_inner_train_steps = 1
    args.meta_learner.args.nb_inner_train_steps = 2
    args.meta_learner.args.nb_inner_train_steps = 4
    args.meta_learner.args.nb_inner_train_steps = 8
    args.meta_learner.args.nb_inner_train_steps = 16
    # args.meta_learner.args.nb_inner_train_steps = 32
    print(f'==> {args.meta_learner.args.nb_inner_train_steps=}')

    # - get dataloaders and overwrites so data analysis runs as we want
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)
    # meta_dataloader = meta_train_dataloader
    meta_dataloader = meta_val_dataloader
    # meta_dataloader = meta_test_dataloader
    print(f'{meta_dataloader.batch_size=}')
    print(f'{meta_dataloader.batch_size=}')

    # - layers to do analysis on
    args.include_final_layer_in_lst = True
    args.layer_names = uutils.torch_uu.get_layer_names_to_do_sim_analysis_fc(args, include_final_layer_in_lst=args.include_final_layer_in_lst)
    # args.layer_names = get_layer_names_to_do_sim_analysis_bn(args, include_final_layer_in_lst=args.include_final_layer_in_lst)
    # args.layer_names = get_layer_names_to_do_sim_analysis_relu(args, include_final_layer_in_lst=args.include_final_layer_in_lst)
    print(args.layer_names)

    # compute functional difference
    args.track_higher_grads = False  # set to false only during meta-testing, but code sets it automatically only for meta-test
    # args.track_higher_grads = True  # set to false only during meta-testing, but code sets it automatically only for meta-test

    # -- start analysis
    print('---------- start analysis ----------')
    print(f'{args.num_workers=}')
    print(f'-->{args.meta_batch_size_eval=}')
    print(f'-->{args.num_its=}')
    print(f'-->{args.nb_inner_train_steps=}')
    print(f'-->{args.metrics_as_dist=}')
    bar_it = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
    # bar_it = uutils.get_good_progressbar(max_value=args.num_its)
    args.it = 1
    halt: bool = False
    while not halt:
        for batch_idx, batch in enumerate(meta_dataloader):
            print(f'it = {args.it}')
            spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

            meta_eval_loss, meta_eval_acc = args.meta_learner(spt_x, spt_y, qry_x, qry_y)
            # diffs_qry = args.meta_learner.functional_difference(spt_x, spt_y, qry_x, qry_y, type_diff='use_qry')
            # dists: dict = args.meta_learner.compute_functional_difference(spt_x, spt_y, qry_x, qry_y, parallel=args.sim_compute_parallel)
            # mean_layer_wise_sim, std_layer_wise_sim, mean_summarized_rep_sim, std_summarized_rep_sim = torch_uu.summarize_similarities(args, sims)

            # -- log it stats
            log_sim_to_check_presence_of_feature_reuse(args, args.it, spt_x, spt_y, qry_x, qry_y, force_log=True, parallel=args.sim_compute_parallel, show_layerwise_sims=args.show_layerwise_sims)

            # - break
            halt: bool = args.it >= args.num_its - 1
            if halt:
                break
            args.it += 1

    # - done!
    print(f'----> {meta_eval_loss=} {meta_eval_acc=}')
    print(f'time_passed_msg = {uutils.report_times(start)}')

    # - wandb
    if is_lead_worker(args.rank) and args.log_to_wandb:
        import wandb
        wandb.finish()

# REPORT TIME
if __name__ == '__main__':
    main_run_expt()
print('--> Success Done! (python print) \a')
