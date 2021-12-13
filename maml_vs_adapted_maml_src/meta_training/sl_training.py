from tqdm import tqdm

import torch

from uutils.torch_uu import accuracy
# from uutils.torch_uu import log_2_tb

from meta_learning.training.meta_training import meta_eval
from meta_learning.training.meta_training import store_check_point
from uutils.torch_uu.tensorboard import log_2_tb_supervisedlearning


def save_check_point_sl(args, n_epoch, model, opt, ckpt_file='ckpt_file.pt'):
    tb = args.tb  # remove and put back tb otherwise toch.save/pickle doesn't work
    args.tb = None
    torch.save({'args': args,
                'state_dict': model.state_dict(),
                'model': model,
                'n_epoch': n_epoch,
                'optimizer': opt,
                'optimizer_dict': opt.state_dict()},
                args.current_logs_path / ckpt_file)
    args.tb = tb  # remove and put back tb otherwise toch.save/pickle doesn't work

def checkpoint_wrt_val_loss(args, n_epoch, model, opt, val_sl_loader, check_point=True):
    """
    This function if checkpoint is true always checkpoints the current model.
    In addition, if the validation loss improved, then that model is also checkpointed seperately.
    """
    # always checkpoint when this function is called (this function is meant to be called only in the val if statement)
    if check_point:
        save_check_point_sl(args, n_epoch, model, opt)
    # do val
    acc_sl, acc5, loss_sl = validate(args, model, val_sl_loader, iter_limit=args.eval_iters)
    loss_std, acc_std = 0.0, 0.0
    # if the current model has a better val loss then checkpoint it
    if loss_sl < args.best_loss:
        args.best_loss = loss_sl
        if check_point:
            save_check_point_sl(args, n_epoch, model, opt, 'ckpt_file_best_loss.pt')
    return acc_sl, acc_std, loss_sl, loss_std

def supervised_train(args, base_model, opt, train_sl_loader, val_sl_loader,
                     meta_learner,
                     meta_valloader):
    # args.best_acc = -1
    args.best_loss = float('inf')
    import warnings
    warnings.simplefilter("ignore")
    # with tqdm(train_sl_loader, total=args.epochs) as pbar_train_sl_loader:
    with tqdm(range(args.train_iters), total=args.train_iters) as pbar_epochs:
        for epoch in pbar_epochs:
            # do 1 epoch on entire data set with SL train
            train_acc, _, train_loss = train(args, base_model, opt, train_sl_loader)

            # anneal learning rate after epoch is done
            args.scheduler.step()

            # store stats
            if epoch % args.log_train_freq == 0 or epoch+1 >= args.train_iters:
                args.logger.log_batch_info(args, loss=train_loss, acc=train_acc, phase='train')  # doesn't log to a log file, store stats in logger obj
                args.logger.loginfo(f"-->phase: train: [e=epoch={epoch}], "
                                    f"train loss: {float(train_loss)}, "
                                    f"train acc: {float(train_acc)}")
                args.logger.save_stats_to_json_file()
                if args.tb:
                    log_2_tb_supervisedlearning(args, epoch, train_loss, train_acc, args.split)

            # store eval stats
            if epoch % args.log_val_freq == 0 or epoch+1 >= args.train_iters:
                acc_sl, acc_std, loss_sl, loss_std = checkpoint_wrt_val_loss(args, epoch, base_model, opt, val_sl_loader, check_point=True)
                # how is it doing in SL
                args.logger.loginfo(f"-->phase: val, : [e=epoch={epoch}], "
                                    f"val loss: {float(loss_sl)} +- {loss_std}, "
                                    f"val acc: {float(acc_sl)} +- {acc_std}")
                # how is it doing in meta-learning
                acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, meta_learner, meta_valloader, iter_limit=args.eval_iters)
                args.logger.loginfo(f"-->phase: meta-val, : [e=epoch={epoch}], "
                                    f"val loss: {float(loss_mean)} +- {loss_std}, "
                                    f"val acc: {float(acc_mean)} +- {acc_std}")
                if args.tb:
                    log_2_tb_supervisedlearning(args, epoch, loss_mean, acc_mean, args.split)
    return acc_mean, acc_std, loss_mean, loss_std

def train(args, model, opt, train_loader):
    """One epoch training"""
    model.train()

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # end = time.time()
    for idx, (input, target, _) in enumerate(train_loader):
        # data_time.update(time.time() - end)
        # input = input.to(args.device)
        # target = target.to(args.device)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        # st()
        output = model(input)
        loss = args.criterion(output, target)

        # st()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        # top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        opt.zero_grad()
        loss.backward()
        # gradient_clip(args, outer_opt)  # do gradient clipping: * If ‖g‖ ≥ c Then g := c * g/‖g‖
        opt.step()

    #     # ===================meters=====================
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #
    #     # tensorboard logger
    #     pass
    #
    #     # print info
    #     if idx % opt.print_freq == 0:
    #         print('Epoch: [{0}][{1}/{2}]\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #               'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #               'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #             epoch, idx, len(train_loader), batch_time=batch_time,
    #             data_time=data_time, loss=losses, top1=top1, top5=top5))
    #         sys.stdout.flush()
    #
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #       .format(top1=top1, top5=top5))
    return acc1, acc5, loss
    #return top1.avg, losses.avg

def validate(args, model, val_loader, iter_limit=2):
    """One epoch validation (up to iter limit) """
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # end = time.time()
        for idx, (input, target, _) in enumerate(val_loader):
            # input = input.to(args.device)
            # target = target.to(args.device)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = args.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), input.size(0))
            # top1.update(acc1[0], input.size(0))
            # top5.update(acc5[0], input.size(0))

        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()
        #
        #     if idx % opt.print_freq == 0:
        #         print('Test: [{0}/{1}]\t'
        #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #               'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #               'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #                idx, len(val_loader), batch_time=batch_time, loss=losses,
        #                top1=top1, top5=top5))
        #
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
            if idx + 1 >= iter_limit:
                break

    return acc1, acc5, loss
    # return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    pass
