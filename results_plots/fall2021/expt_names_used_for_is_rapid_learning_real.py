
# debug init
# logs/logs_Nov27_15-31-36_jobid_444362.iam-pbs/ckpt_file.pt
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov17_13-57-11_jobid_416472.iam-pbs/ckpt_file.pt').expanduser()  # debugging ckpt

# jobs path for ML + loss vs STD (overfitted)
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-31-36_jobid_444362.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov30_03-38-25_jobid_444368.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-36-34_jobid_444374.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-39-49_jobid_444380.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# # jobs path for ML + loss vs STD (best)
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-31-36_jobid_444362.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov30_03-38-25_jobid_444368.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-36-34_jobid_444374.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-39-49_jobid_444380.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (overfitted sigmoid)
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-32-18_jobid_444365.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-35-48_jobid_444371.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (best sigmoid)
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-32-18_jobid_444365.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-35-48_jobid_444371.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (relu)
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file.pt').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-36-58_jobid_444376.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# maml.args.nb_inner_train_steps = 32
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# jobs path for ML + loss vs STD (sigmoid)
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file.pt').expanduser()
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov27_15-37-18_jobid_444377.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml = ckpt['meta_learner']
# maml.args.nb_inner_train_steps = 0
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# sinusoid experiment
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Nov23_11-26-20_jobid_438708.iam-pbs/ckpt_file.pt').expanduser()
# args.data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# ckpt = torch.load(path_2_init)
# maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
# args.num_workers = 0
# # maml.args.inner_lr = 0.01
# maml.args.nb_inner_train_steps = 0
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_sine_dataloader(args)

# tiny std1 for task
# path_2_init = Path('/Users/brando/data_fall2020_spring2021/logs/logs_Dec04_16-16-31_jobid_446021.iam-pbs/ckpt_file_best_loss.pt').expanduser()
# ckpt = torch.load(path_2_init)
# maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
# args.data_path = Path('/Users/brando/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.01_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
# args.num_workers = 0
# # maml.args.inner_lr = 0.01
# # maml.args.nb_inner_train_steps = 0
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)

# tiny std1 for task
# path_2_init = Path('~/data_fall2020_spring2021/logs/logs_Dec30_15-48-29_jobid_2387/ckpt_file_best_loss.pt').expanduser()
# ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
# maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
# args.data_path = Path('~/data_fall2020_spring2021/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.01_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
# args.num_workers = 0
# # maml.args.inner_lr = 0.01
# # maml.args.nb_inner_train_steps = 0
# args.meta_batch_size_train = 200; args.meta_batch_size_eval = args.meta_batch_size_train
# meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_randfnn_dataloader(args)
# print(meta_val_dataloader.batch_size)

# large std1 for task
# path_2_init = Path('/logs/logs_Dec30_15-53-29_jobid_2393/ckpt_file_best_loss.pt').expanduser()
# ckpt = torch.load(path_2_init, map_location=torch.device('cpu'))
# maml, args = ckpt['meta_learner'], ckpt['meta_learner'].args
# args.data_path = Path('/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
