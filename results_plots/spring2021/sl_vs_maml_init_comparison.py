"""
Technical comments

Comparison between models trained with different number of classes:
    - for now since SL vs MAML are trained with different number of classes, we will not compare the final layer where
    the probability of a class is calculated (due to different sizes). Only comparing same size layers. i.e. the
    representation layers (1 to L-1 and not 1 to L)

"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore")


import sys
import time
from argparse import Namespace
from pprint import pprint

import torch

from pathlib import Path

from meta_learning.base_models.resnet_rfs import resnet12, resnet18
from meta_learning.meta_learners.pretrain_convergence import FitFinalLayer
from meta_learning.training.meta_training import process_meta_batch, meta_eval
from progressbar import ProgressBar
from uutils.torch import get_distance_of_inits, check_mdl_in_single_gpu
from uutils.torch.dataloaders import get_rfs_sl_dataloader, process_batch_sl, get_miniimagenet_dataloaders_torchmeta

from pdb import set_trace as st

def parse_args():
    import argparse
    from uutils import create_logs_dir_and_load, load_cluster_jobids_to
    from datetime import datetime

    parser = argparse.ArgumentParser()

    # experimental setup
    parser.add_argument('--debug', action='store_true', help='if debug')
    # parser.add_argument('--data_path', type=str, default=Path("~/data/lasse_datasets_coq/all-sexpr.feat").expanduser())
    parser.add_argument('--log_root', type=str, default=Path('/logs/').expanduser())

    # for compute stats
    parser.add_argument('--iters', type=int, default=4,
                        help='number of iterations for evaluation. Each it selects a meta-batch of taks.')
    parser.add_argument('--meta_batch_size', type=int, default=8,
                        help='meta-batch-size is the number of tasks used in one iteration')
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='meta-batch-size is the number of tasks used in one iteration')
    # Other
    parser.add_argument('--num_workers', type=int, default=4, help='value zero to avoid pycharm errors')
    parser.add_argument('--device', type=str, default='detect_automatically',
                        help='options are gpu or cpu otherwise it detects automatically')

    # parse arguments
    args = parser.parse_args()

    if args.debug:
        args.num_workers = 0

    # save number of cpus
    args.cpu_count = torch.multiprocessing.cpu_count()
    if args.device != 'detect_automatically':
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    args.test_batch_size = args.batch_size
    return args

def get_resnet12_rfs_maml(args):
    path_2_init = Path('~/data/logs/logs_Nov13_18-07-41_jobid_858/ckpt_file.pt').expanduser()
    # path_2_init = Path('~/data/logs/logs_Nov25_16-47-55_jobid_1302/ckpt_file.pt').expanduser()
    if str(args.device) == 'cpu':
        ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(path_2_init)
    maml = ckpt['meta_learner']
    model = maml.base_model
    # second overwrites the first so my command line arguments wins
    args = Namespace(**{**vars(ckpt['meta_learner'].args), **vars(args)})
    args.k_eval = 30
    # maml.args.inner_lr = 0.01
    # maml.args.nb_inner_train_steps = 0
    model.to(args.device)
    model.eval()
    #
    ffl = FitFinalLayer(args, model, target_type='classification', classifier='LR')
    return model, maml, ffl, args

def get_resnet12_rfs_sl_original(args, mdl_type='simple'):
    # get path to model
    if mdl_type == 'simple':  # not distilled model
        path_2_init = Path('~/data/rfs_checkpoints/mini_simple.pth').expanduser()
        # path_2_init = Path("~/data/logs/logs_Mar06_11-21-31_jobid_4_pid_24692/ckpt_file_best_loss.pt").expanduser()
    elif mdl_type == 'my_reproduction':  # not distilled model
        path_2_init = Path("~/data/logs/logs_Mar06_11-21-31_jobid_4_pid_24692/ckpt_file_best_loss.pt").expanduser()
    else:
        path_2_init = Path('~/data/rfs_checkpoints/mini_distilled.pth').expanduser()

    # loack entire checkpoint file to de device
    if str(args.device) == 'cpu':
        ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(path_2_init)

    # load model to device
    if mdl_type == 'simple' or mdl_type == 'distilled':  # not distilled model
        model_dict = ckpt['model']
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)
        model.load_state_dict(model_dict)
    elif mdl_type == 'my_reproduction':
        model = ckpt['model']

    model.to(args.device)
    model.eval()
    #
    ffl = FitFinalLayer(args, model, target_type='classification', classifier='LR')
    return model, ffl

def main_sl_maml_get_meta_vals():
    print()
    print('---- running ----')
    print(f'is cuda available: {torch.cuda.is_available()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    args = parse_args()
    pprint(args, sort_dicts=True)

    # load models
    f_maml, maml, ffl_maml, args = get_resnet12_rfs_maml(args)
    f_sl, ffl_rfs = get_resnet12_rfs_sl_original(args, mdl_type='my_reproduction')
    # f_sl, ffl_rfs = get_resnet12_rfs_sl_original(args)
    print(check_mdl_in_single_gpu(f_sl))
    print(check_mdl_in_single_gpu(f_maml))

    # get dataloaders
    train_sl_loader, val_sl_loader, meta_valloader = get_rfs_sl_dataloader(args)
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_miniimagenet_dataloaders_torchmeta(args)

    # report meta-val using LRF
    print('\n-- meta-eval accs & losses --')
    print(f'{meta_train_dataloader.batch_size=}')
    iter_limit = args.iters
    acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, ffl_rfs, meta_val_dataloader, iter_limit)
    print(f'-> rfs ffl accs: {acc_mean=}, {acc_std=}, {loss_mean=}, {loss_std=}')
    acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, ffl_rfs, meta_test_dataloader, iter_limit)
    print(f'-> rfs ffl accs: {acc_mean=}, {acc_std=}, {loss_mean=}, {loss_std=}')
    acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, ffl_rfs, meta_valloader, iter_limit)
    print(f'-> rfs ffl accs: {acc_mean=}, {acc_std=}, {loss_mean=}, {loss_std=}')
    # acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, ffl_maml, meta_val_dataloader, iter_limit)
    # print(f'-> maml ffl accs: {acc_mean=}, {acc_std=}, {loss_mean=}, {loss_std=}')
    print()

def main_sl_vs_maml_miniimagenet():
    print()
    print('---- running ----')
    print(f'is cuda available: {torch.cuda.is_available()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    args = parse_args()

    # load models
    f_maml, maml, ffl_maml, args = get_resnet12_rfs_maml(args)
    f_sl, ffl_rfs = get_resnet12_rfs_sl_original(args)
    print(check_mdl_in_single_gpu(f_sl))
    print(check_mdl_in_single_gpu(f_maml))

    # get dataloaders
    meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_miniimagenet_dataloaders_torchmeta(args)
    # train_sl_loader, val_sl_loader, meta_valloader = get_rfs_sl_dataloader(args)

    # do comparison
    # bar = ProgressBar(max_value=len(args.iters))
    # sys.exit()
    st()
    bar = ProgressBar(max_value=args.epochs)
    for epoch, (batch_x, batch_y, _) in enumerate(train_sl_loader):
        batch_x, batch_y = process_batch_sl(args, batch=[batch_x, batch_y])
        ds = get_distance_of_inits(args, batch=[batch_x, batch_y], f1=f_sl, f2=f_maml)
        print(ds)
        bar.update(epoch)
    print("done MAML vs SL")

if __name__ == '__main__':
    start = time.time()
    # main_sl_vs_maml_miniimagenet()
    main_sl_maml_get_meta_vals()
    print(f'\ntime duration: {time.time() - start} seconds')