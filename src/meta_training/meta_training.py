import warnings
from argparse import Namespace

import torch
from torch import nn

import uutils
from uutils.torch_uu import gradient_clip, ned_torch, process_meta_batch
from uutils.torch_uu.distributed import is_lead_worker, get_model_from_ddp
from uutils.torch_uu.tensorboard import log_2_tb, log_2_tb_metalearning, log_2_tb_supervisedlearning

def save_for_meta_learning(args: Namespace, ckpt_filename: str = 'ckpt.pt'):
    """
    ref:
        - https://stackoverflow.com/questions/70129895/why-is-it-not-recommended-to-save-the-optimizer-model-etc-as-pickable-dillable
    """
    if is_lead_worker(args.rank):
        import dill
        args.logger.save_current_plots_and_stats()
        # - ckpt
        assert uutils.xor(args.training_mode == 'epochs', args.training_mode == 'iterations')
        f: nn.Module = get_model_from_ddp(args.base_model)
        # pickle vs torch.save https://discuss.pytorch.org/t/advantages-disadvantages-of-using-pickle-module-to-save-models-vs-torch-save/79016
        args_pickable: Namespace = uutils.make_args_pickable(args)
        torch.save({'training_mode': args.training_mode,  # assert uutils.xor(args.training_mode == 'epochs', args.training_mode == 'iterations')
                    'it': args.it,
                    'epoch_num': args.epoch_num,

                    'args': args_pickable,  # some versions of this might not have args!

                    'meta_learner': args.meta_learner,
                    'meta_learner_str': str(args.meta_learner),  # added later, to make it easier to check what optimizer was used

                    'f': f,
                    'f_state_dict': f.state_dict(),  # added later, to make it easier to check what optimizer was used
                    'f_str': str(f),  # added later, to make it easier to check what optimizer was used
                    # 'f_modules': f._modules,
                    # 'f_modules_str': str(f._modules),

                    'outer_opt': args.outer_opt,  # added later, to make it easier to check what optimizer was used
                    'outer_opt_state_dict': args.outer_opt.state_dict(),  # added later, to make it easier to check what optimizer was used
                    'outer_opt_str': str(args.outer_opt),  # added later, to make it easier to check what optimizer was used

                    'scheduler_str': str(args.scheduler),
                    'scheduler': args.scheduler
                    },
                   pickle_module=dill,
                   f=args.log_root / ckpt_filename)

def log_sim_to_check_presence_of_feature_reuse(args: Namespace,
                                               it: int,

                                               spt_x, spt_y, qry_x, qry_y,  # these are multiple tasks

                                               log_freq_for_detection_of_feature_reuse: int = 3,

                                               force_log: bool = False,
                                               parallel: bool = False,
                                               iter_tasks=None,
                                               log_to_wandb: bool = False,
                                               show_layerwise_sims: bool = True
                                               ):
    """
    Goal is to see if similarity is small s <<< 0.9 (at least s < 0.8) since this suggests that
    """
    import wandb
    import uutils.torch_uu as torch_uu
    from pprint import pprint
    from uutils.torch_uu import summarize_similarities
    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'
    sim_or_dist: str = 'sim'
    if hasattr(args, 'metrics_as_dist'):
        sim_or_dist: str = 'dist' if args.metrics_as_dist else sim_or_dist
    total_its: int = args.num_empochs if args.training_mode == 'epochs' else args.num_its

    if (it == total_its - 1 or force_log) and is_lead_worker(args.rank):
    # if (it % log_freq_for_detection_of_feature_reuse == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        if hasattr(args, 'metrics_as_dist'):
            sims = args.meta_learner.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names, parallel=parallel, iter_tasks=iter_tasks, metric_as_dist=args.metrics_as_dist)
        else:
            sims = args.meta_learner.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, args.layer_names, parallel=parallel, iter_tasks=iter_tasks)
        mean_layer_wise_sim, std_layer_wise_sim, mean_summarized_rep_sim, std_summarized_rep_sim = summarize_similarities(args, sims)

        # -- log (print)
        args.logger.log(f' \n------ {sim_or_dist} stats: {it_or_epoch}={it} ------')
        # - per layer
        # if show_layerwise_sims:
        print(f'---- Layer-Wise metrics ----')
        print(f'mean_layer_wise_{sim_or_dist} (per layer)')
        pprint(mean_layer_wise_sim)
        print(f'std_layer_wise_{sim_or_dist} (per layer)')
        pprint(std_layer_wise_sim)

        # - rep sim
        print(f'---- Representation metrics ----')
        print(f'mean_summarized_rep_{sim_or_dist} (summary for rep layer)')
        pprint(mean_summarized_rep_sim)
        print(f'std_summarized_rep_{sim_or_dist} (summary for rep layer)')
        pprint(std_summarized_rep_sim)
        args.logger.log(f' -- sim stats : {it_or_epoch}={it} --')

        # error bars with wandb: https://community.wandb.ai/t/how-does-one-plot-plots-with-error-bars/651
        # - log to wandb
        # if log_to_wandb:
        #     if it == 0:
        #         # have all metrics be tracked with it or epoch (custom step)
        #         #     wandb.define_metric(f'layer average {metric}', step_metric=it_or_epoch)
        #         for metric in mean_summarized_rep_sim.keys():
        #             wandb.define_metric(f'rep mean {metric}', step_metric=it_or_epoch)
        #     # wandb.log per layer
        #     rep_summary_log = {f'rep mean {metric}': sim for metric, sim in mean_summarized_rep_sim.items()}
        #     rep_summary_log[it_or_epoch] = it
        #     wandb.log(rep_summary_log, commit=True)

def log_train_val_stats(args: Namespace,
                        it: int,

                        train_loss: float,
                        train_acc: float,

                        valid,

                        bar,

                        log_freq: int = 10,
                        ckpt_freq: int = 50,
                        mdl_watch_log_freq: int = 50,
                        force_log: bool = False,  # e.g. at the final it/epoch

                        save_val_ckpt: bool = False,
                        log_to_tb: bool = False,
                        log_to_wandb: bool = False
                        ):
    """
    Log train and val stats where it is iteration or epoch step.

    Note: Unlike save ckpt, this one does need it to be passed explicitly (so it can save it in the stats collector).
    """
    import wandb
    from uutils.torch_uu.tensorboard import log_2_tb_supervisedlearning
    from pprint import pprint
    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'

    # if its
    total_its: int = args.num_empochs if args.training_mode == 'epochs' else args.num_its

    if (it % log_freq == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        # - get eval stats
        val_loss, val_acc, val_loss_std, val_acc_std = valid(args, save_val_ckpt=save_val_ckpt)
        # - log ckpt
        if it % ckpt_freq == 0:
            save_for_meta_learning(args)

        # - save args
        uutils.save_args(args, args_filename='args.json')

        # - update progress bar at the end
        bar.update(it)

        # - print
        args.logger.log('\n')
        args.logger.log(f"{it_or_epoch}={it}: {train_loss=}, {train_acc=}")
        args.logger.log(f"{it_or_epoch}={it}: {val_loss=}, {val_acc=}")

        print(f'{args.it=}')
        print(f'{args.num_its=}')

        # - record into stats collector
        args.logger.record_train_stats_stats_collector(it, train_loss, train_acc)
        args.logger.record_val_stats_stats_collector(it, val_loss, val_acc)
        args.logger.save_experiment_stats_to_json_file()
        args.logger.save_current_plots_and_stats()

        # - log to wandb
        if log_to_wandb:
            if it == 0:
                wandb.define_metric("train loss", step_metric=it_or_epoch)
                wandb.define_metric("train acc", step_metric=it_or_epoch)
                wandb.define_metric("val loss", step_metric=it_or_epoch)
                wandb.define_metric("val val", step_metric=it_or_epoch)
                # if mdl_watch_log_freq == -1:
                #     wandb.watch(args.base_model, args.criterion, log="all", log_freq=mdl_watch_log_freq)
            # - log to wandb
            wandb.log(data={it_or_epoch: it,  # custom step: https://community.wandb.ai/t/how-is-one-suppose-to-do-custom-logging-in-wandb-especially-with-the-x-axis/1400
                            'train loss': train_loss,
                            'train acc': train_acc,
                            'val loss': val_loss,
                            'val acc': val_acc},
                            commit=True)
            # if it == total_its:  # not needed here, only needed for normal SL training
            #     wandb.finish()

        # - log to tensorboard
        if log_to_tb:
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'train')
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'val')



# -- training code

def meta_train_epochs(args, meta_learner, outer_opt, meta_train_dataloader, meta_val_dataloader):
    """ Meant for episodic training """
    args.epochs = args.train_iters
    args.best_acc = -1
    args.best_loss = float('inf')
    with tqdm(range(args.epochs)) as tepochs:
        for epoch in tepochs:
            for batch_idx, batch in enumerate(meta_train_dataloader):
                args.batch_idx = batch_idx; args.epoch = epoch
                spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

                outer_opt.zero_grad()

                meta_train_loss, meta_train_acc = meta_learner(spt_x, spt_y, qry_x, qry_y)

                gradient_clip(args, outer_opt)  # do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
                outer_opt.step()

            # -- log after each epoch
            log_meta_train_info(args, batch_idx, meta_train_loss, meta_train_acc)
            log_meta_val_info(args, batch_idx, meta_learner, outer_opt, meta_val_dataloader)
            # - break
            if batch_idx + 1 >= args.train_iters:
                print('BREAKING')
                break

# - evaluation code

def meta_eval(args: Namespace, training: bool = True, val_iterations: int = 0, save_val_ckpt: bool = True, split: str = 'val') -> tuple:
    """
    Evaluates the meta-learner on the given meta-set.

    ref for BN/eval:
        - tldr: Use `mdl.train()` since that uses batch statistics (but inference will not be deterministic anymore).
        You probably won't want to use `mdl.eval()` in meta-learning.
        - https://stackoverflow.com/questions/69845469/when-should-one-call-eval-and-train-when-doing-maml-with-the-pytorch-highe/69858252#69858252
        - https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        - https://github.com/tristandeleu/pytorch-maml/issues/19
    """
    # - need to re-implement if you want to go through the entire data-set to compute an epoch (no more is ever needed)
    assert val_iterations == 0, f'Val iterations has to be zero but got {val_iterations}, if you want more precision increase (meta) batch size.'
    args.meta_learner.train() if training else args.meta_learner.eval()
    for batch_idx, batch in enumerate(args.dataloaders[split]):
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)

        # Forward pass
        # eval_loss, eval_acc = args.meta_learner(spt_x, spt_y, qry_x, qry_y)
        eval_loss, eval_acc, eval_loss_std, eval_acc_std = args.meta_learner(spt_x, spt_y, qry_x, qry_y, training=training)

        # store eval info
        if batch_idx >= val_iterations:
            break

    if float(eval_loss) < float(args.best_val_loss) and save_val_ckpt:
        args.best_val_loss = float(eval_loss)
        save_for_meta_learning(args, ckpt_filename='ckpt_best_val.pt')
    return eval_loss, eval_acc, eval_loss_std, eval_acc_std
    # return eval_loss, eval_acc
