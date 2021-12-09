#!/home/miranda9/miniconda3/envs/automl-meta-learning/bin/python

import torch
import torch.nn as nn
import torch.optim as optim
# import torch_uu.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm

import numpy as np

import uutils
from uutils import report_times, load_cluster_jobids_to
# TODO check if needed from uutils import collect_content_from_file
from uutils.emailing import send_email
from uutils.logger import Logger
from uutils import get_truly_random_seed_through_os
# from uutils import get_cluster_jobids

from uutils.torch_uu import replace_bn

# from meta_learning.training.meta_training import meta_train, meta_eval, meta_train_epochs
from meta_learning.training.supervised_training import supervised_train

from meta_learning.datasets.mini_imagenet import ImageNet, MetaImageNet
# from datasets.tiered_imagenet import TieredImageNet, MetaTieredImageNet
# from datasets.cifar import CIFAR100, MetaCIFAR100
# from datasets.transform_cfg import transforms_options, transforms_list

from meta_learning.meta_learners.maml_meta_learner import MAMLMetaLearner
from meta_learning.meta_learners.pretrain_convergence import FitFinalLayer

from meta_learning.base_models.resnet_rfs import resnet12, resnet18, resnet24, resnet50, resnet101
from meta_learning.base_models.learner_from_opt_as_few_shot_paper import Learner

from meta_learning.datasets.rand_fc_nn_vec_mu_ls_gen import get_backbone

import os
import subprocess  # for getting githash
from socket import gethostname

from types import SimpleNamespace
from collections import OrderedDict

import random

import time
from datetime import datetime

import pathlib
from pathlib import Path

import json

from uutils.torch_uu.dataloaders import get_rfs_sl_dataloader


def manual_load(args):
    args.debug = False
    load_cluster_jobids_to(args)  # args.jobid = SETVALUE HERE

    args.k_shots = 5
    args.k_eval = 15
    args.n_classes = 5

    args.n_aug_support_samples = 5  # but rfs default is 5...
    args.out_features = 64  # number of classification classes for last layer
    # args.grad_clip_rate = None  # does no gradient clipping if None
    # args.grad_clip_mode = None  # more specific setting of the crad clipping split

    # training & eval hyper-params
    args.trainin_with_epochs = False
    # args.trainin_with_epochs = True

    args.train_iters = 600  # epochs for SL
    args.epochs = args.train_iters
    args.log_train_freq = 5 if not args.debug else 1

    args.eval_iters = 5
    # args.meta_batch_size_eval = 1
    args.test_batch_size = 1  # rfs default is 1
    args.log_val_freq = 5 if not args.debug else 1  # for hyperparam tuning. note: lower the quicker the code.

    # meta-learner params
    # args.meta_learner = 'maml_fixed_inner_lr'
    # args.inner_lr = 1e-1
    # args.nb_inner_train_steps = 4
    # args.track_higher_grads = True  # set to false only during meta-testing, but code sets it automatically only for meta-test
    # args.copy_initial_weights = False  # DONT PUT FALSE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    # args.fo = False
    # args.outer_lr = 1e-1
    # args.outer_lr = 5e-2  # rfs uses 5e-2

    # pff
    args.meta_learner = 'FitFinalLayer'
    # Data-set options
    args.split = "train"
    # args.split = 'val'
    # args.split = "test"
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_8.0_std2_1.0_noise_std0.1/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_16.0_std2_1.0_noise_std0.1').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_32.0_std2_1.0_noise_std0.1/').expanduser()
    # mini-imagenet
    args.data_path = 'miniimagenet'
    # Data loader options
    args.num_workers = 4
    args.pin_memory = False  # it is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True), which enables fast data transfer to CUDA-enabled GPUs. https://pytorch.org/docs/stable/data.html
    # Tensorboard
    # args.tb = False
    args.tb = True
    # Base model
    # args.base_model_mode = 'debug'
    args.base_model_mode = 'child_mdl_from_opt_as_a_mdl_for_few_shot_learning_paper'
    # args.base_model_mode = 'resnet12_rfs'
    # args.base_model_mode = 'resnet18_rfs'
    # args.base_model_mode = 'resnet24_rfs'
    # args.base_model_mode = 'resnet50_rfs'
    # args.base_model_mode = 'resnet101_rfs'
    # args.base_model_mode = 'resnet18'
    # args.base_model_mode = 'resnet50'
    # args.base_model_mode = 'resnet101'
    # args.base_model_mode = 'resnet152'
    # args.base_model_mode = 'rand_init_true_arch'
    # args.base_model_mode = 'f_avg'
    # args.base_model_mode = 'f_avg_add_noise'
    # args.base_model_mode = 'custom_synthetic_backbone'
    # args.base_model_mode = Path('~/data/logs/logs_Sep29_13-05-52_jobid_383794.iam-pbs/ckpt_file.pt').expanduser()
    # args.base_model_mode = Path('/home/miranda9/data/logs/logs_Nov05_15-12-13_jobid_1729/ckpt_file.pt')
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov13_13-23-04_jobid_473828/ckpt_file.pt'
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov13_15-58-43_jobid_473871/ckpt_file.pt'
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov13_15-11-37_jobid_473870/ckpt_file.pt'
    # args.base_model_mode = Path(args.base_model_mode).expanduser()
    # Logger stuff
    args.log_root = Path('~/data/logs/').expanduser()
    # email option
    args.mail_user = 'brando.science@gmail.com'
    args.pw_path = Path('~/pw_app.config.json').expanduser()
    args.save_ckpt = True
    # Set random seed
    args.seed = None  # None selects a (truly) random seed from os
    # assert (args.debug == False)
    return args

def main(args):
    print('-------> Inside Experiment Code <--------')
    print(f"\n---> hostname: {gethostname()}, current_time: {datetime.now().strftime('%b%d_%H-%M-%S')}")
    args.hostname = gethostname()

    # -- Set up logger
    uutils.save_args(args, )

    # -- Device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.gpu_name = args.device
    if args.device.type != 'cpu':  # if not CPU then it's GPU/cuda
        if not torch.cuda.is_available():  # must be using gpu/cuda
            raise RuntimeError("GPU unavailable.")
        # For reproducibitity: https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        args.gpu_name = torch.cuda.get_device_name(0)
        print(f'\nargs.gpu_name = {args.gpu_name}\n')
    except:
        args.gpu_name = args.device
    print(f'\nargs.device = {args.device}\n')

    # -- Set upt reproducibility ref: https://pytorch.org/docs/stable/notes/randomness.html
    if args.seed is None:  # is args.seed is None then select a random intiger
        # For reproducibility: set random seed for all libraries we are using
        # args.seed = random.randint(0, 1e3)
        args.seed = get_truly_random_seed_through_os()  # to later: https://github.com/brando90/automl-meta-learning/issues/63
        print(f'>>> Selected random seed: {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -- NN Model
    print(f'--> args.base_model_model: {args.base_model_mode}')
    if args.base_model_mode == 'debug':
        from uutils.torch_uu.models.custom_layers import Flatten
        model = nn.Sequential(OrderedDict([
            ('features', nn.Sequential(OrderedDict([('flatten', Flatten())]))),
            ('cls', torch.nn.Linear(in_features=84*84*3, out_features=args.out_features, bias=True))
        ]))
        args.base_model = nn.Sequential(OrderedDict([('model', model)]))
    elif args.base_model_mode == 'child_mdl_from_opt_as_a_mdl_for_few_shot_learning_paper':
        args.batch_size = 1028
        args.bn_momentum = 0.95
        args.bn_eps = 1e-3
        args.grad_clip_mode = 'clip_all_together'
        args.image_size = 84
        args.base_model = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum, n_classes=args.n_classes).to(args.device)
        in_features = args.base_model.model.cls.in_features
        args.base_model.model.cls = torch.nn.Linear(in_features=in_features, out_features=args.out_features, bias=True)
    elif args.base_model_mode == 'resnet12_rfs':
        args.batch_size = 64
        args.base_model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.out_features).to(args.device)
    elif args.base_model_mode == 'resnet18_rfs':
        args.batch_size = 64
        args.base_model = resnet18(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.out_features).to(args.device)
    elif args.base_model_mode == 'resnet24_rfs':
        args.batch_size = 64
        args.base_model = resnet24(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.out_features).to(args.device)
    elif args.base_model_mode == 'resnet50_rfs':
        args.batch_size = 64
        args.base_model = resnet50(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.out_features).to(args.device)
    elif args.base_model_mode == 'resnet101_rfs':
        args.batch_size = 64
        args.base_model = resnet101(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.out_features).to(args.device)
    elif args.base_model_mode == 'resnet18':
        args.batch_size = 64
        args.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        # replace_bn(args.base_model, 'model')
        args.base_model.fc = torch.nn.Linear(in_features=512, out_features=args.out_features, bias=True)
    elif args.base_model_mode == 'resnet50':
        args.batch_size = 64
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.out_features, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'resnet101':
        args.batch_size = 64
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
        replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.out_features, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'resnet152':
        args.batch_size = 64
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)
        replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.out_features, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'custom_synthetic_backbone':
        target_f_name = 'fully_connected_NN_with_BN'
        # params for backbone
        Din, Dout = 1, 1
        # H = 15*200
        H = 2*3
        # 10 layers, 9 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 9 layers, 8 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 8 layers, 7 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 7 layers, 6 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 6 layers, 5 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 5 layers, 4 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 4 layers, 3 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, Dout)]
        # 3 layers, 2 hidden layers
        hidden_dim = [(Din, H), (H, H), (H, Dout)]
        print(f'# of hidden layers = {len(hidden_dim) - 1}')
        print(f'total layers = {len(hidden_dim)}')
        section_label = [1] * (len(hidden_dim) - 1) + [2]
        task_gen_params = {
            'metaset_path': None,
            'target_f_name': target_f_name,
            'hidden_dim': hidden_dim,
            'section_label': section_label,
            'Din': Din, 'Dout': Dout, 'H': H
        }
        args.base_model = get_backbone(task_gen_params)
        args.task_gen_params = task_gen_params
    elif type(args.base_model_mode) is pathlib.PosixPath:
        # db = torch_uu.load(str(args.resume_ckpt_path))
        db = torch.load(str(args.base_model_mode))
        # meta_learner = db['meta_learner']
        args.base_model = db['f']
        # in case loading directly doesn't work
        # modules = eval(db['f_modules_str'])
        # args.base_model = torch_uu.nn.Sequential(modules)
        # f_state_dict = db['f_state_dict']
        # args.base_model.load_state_dict(f_state_dict)
        print('RUNNING FROM CHECKPOINT')
        print('RUNNING FROM CHECKPOINT')
    else:
        raise ValueError(f'Not Implemented: args.base_model_mode = {args.base_model_mode}')
    args.base_model.to(args.device)

    # -- Outer optimizer
    args.outer_lr = 1e-4
    args.outer_opt = torch.optim.Adam(args.base_model.parameters(), lr=args.outer_lr, weight_decay=5e-4)
    # args.outer_opt = optim.SGD(args.base_model.parameters(), lr=args.outer_lr, momentum=0.9, weight_decay=5e-4)

    # -- Meta-learner
    if args.meta_learner == 'maml_fixed_inner_lr':
        args.grad_clip_rate = None
        meta_learner = MAMLMetaLearner(args, args.base_model, fo=args.fo, lr_inner=args.inner_lr)
    elif args.meta_learner == 'meta_lstm':
        # args.grad_clip_rate = 0.25
        # meta_learner = MetaTrainableLstmOptimizer(args.base_model, args.lr_inner)
        # Gradient clipping params
        # args.grad_clip_rate = 0.25
        # args.grad_clip_mode = 'clip_all_together'
        # args.grad_clip_mode = 'clip_all_seperately'
        raise ValueError('Meta-lstm not impemented')
        raise ValueError('Meta-lstm not impemented')
    elif args.meta_learner == "FitFinalLayer":
        meta_learner = FitFinalLayer(args, args.base_model)
        args.inner_opt_name = 'PFF'
    else:
        raise ValueError(f"Invalid trainable opt: {args.meta_learner}")

    # -- Data set & data loaders
    if 'miniimagenet' in str(args.data_path):
        train_sl_loader, val_sl_loader, meta_valloader = get_rfs_sl_dataloader(args)
    elif 'fully_connected' in str(args.data_path.name):
        args.target_type = 'regression'
        args.criterion = nn.CrossEntropyLoss()
        # meta_learner.regression()
        # # get data
        # dataset_train = RandFNN(args.data_path, 'train')
        # dataset_val = RandFNN(args.data_path, 'val')
        # dataset_test = RandFNN(args.data_path, 'test')
        # # get meta-sets
        # metaset_train = ClassSplitter(dataset_train,
        #                               num_train_per_class=args.k_shots,
        #                               num_test_per_class=args.k_eval,
        #                               shuffle=True)
        # metaset_val = ClassSplitter(dataset_val, num_train_per_class=args.k_shots,
        #                             num_test_per_class=args.k_eval,
        #                             shuffle=True)
        # metaset_test = ClassSplitter(dataset_test, num_train_per_class=args.k_shots,
        #                              num_test_per_class=args.k_eval,
        #                              shuffle=True)
        # # get meta-dataloader
        # meta_train_dataloader = BatchMetaDataLoader(metaset_train,
        #                                             batch_size=args.meta_batch_size_train,
        #                                             num_workers=args.num_workers)
        # meta_val_dataloader = BatchMetaDataLoader(metaset_val,
        #                                           batch_size=args.meta_batch_size_eval,
        #                                           num_workers=args.num_workers)
        # meta_test_dataloader = BatchMetaDataLoader(metaset_test,
        #                                            batch_size=args.meta_batch_size_eval,
        #                                            num_workers=args.num_workers)
        pass
    else:
        raise ValueError(f'Not such task: {args.data_path}')

    # -- Scheduler
    # scheduler used in rfs
    args.lr_decay_rate = 0.1
    eta_min = args.outer_lr * (args.lr_decay_rate ** 3)
    args.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.outer_opt, args.epochs, eta_min, -1)

    # -- tensorboard
    if args.tb:
        print(f'Is tensorboard being used? {args.tb}')
        log_dir = args.current_logs_path / 'tb/'
        args.tb = SummaryWriter(str(log_dir))

    # -- Run experiment
    print(f'About to start training with args: {args}')
    if args.split == 'train':
        print('--------------------- TRAIN: SL ------------------------')
        # args.expt_results = meta_train(args, meta_learner, args.outer_opt, meta_train_dataloader, meta_val_dataloader)
        args.expt_results = supervised_train(args,
                                             args.base_model,
                                             args.outer_opt,
                                             train_sl_loader,
                                             val_sl_loader,
                                             meta_learner,
                                             meta_valloader)
        args.logger.save_stats_to_json_file()
    else:
        raise ValueError(f'Value error: args.split = {args.split}, is not a valid split.')

    # End of experiment code
    print('---> Done with main')
    return args


if __name__ == "__main__":
    # -- start script
    start = time.time()
    print(f'-----> device = {torch.device("cuda" if torch.cuda.is_available() else "cpu")}\n')
    args = manual_load(SimpleNamespace())
    main(args)
    args.time_passed_msg, _, _, _ = report_times(start)
    print(f'\n---> hostname: {gethostname()}\n tb = {args.tb}')
    print('--> final DONE <--\a')
    # Experiments last info
    acc_mean, acc_std, loss_mean, loss_std = args.expt_results
    expt_msg = f'''
--
jobid_{args.jobid}
hostname {gethostname()}
device {args.device}
--
base_model_mode: {args.base_model_mode}
meta_learner: {args.meta_learner}
{args.data_path}
splt: {args.split}
{args.time_passed_msg} 
nb epochs: {args.train_iters}, epochs_eval: {args.eval_iters}, nb_inner_train_steps: {args.nb_inner_train_steps}, inner_lr: {args.inner_lr}
eval loss: {loss_mean} +- {loss_std} 
eval acc: {acc_mean} +- {acc_std} 
inner_opt: {args.inner_opt_name}
outer_opt: {args.outer_opt}
    '''
    print(expt_msg)
    with open(args.current_logs_path /"quick_expt.txt", "w") as text_file:
        text_file.write(expt_msg)
    send_email(subject='QUICK EXPT', message=expt_msg, destination=args.mail_user, password_path=args.pw_path)
    print('DONE \a')

