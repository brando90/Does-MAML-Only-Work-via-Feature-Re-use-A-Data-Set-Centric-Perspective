import torch
import torch.nn as nn
from torch.multiprocessing import Pool
from torch.optim.optimizer import required
from torch.optim import Optimizer as Optimizer

import higher
from higher.optim import _add
from higher.optim import DifferentiableOptimizer
from higher.optim import _GroupedGradsType

import uutils
from uutils.torch_uu import functional_diff_norm, ned_torch, r2_score_from_torch, calc_accuracy_from_logits, \
    normalize_matrix_for_similarity
from uutils.torch_uu import tensorify

import numpy as np

Spt_x, Spt_y, Qry_x, Qry_y = torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
Task = tuple[Spt_x, Spt_y, Qry_x, Qry_y]
Batch = list

##

class EmptyOpt(Optimizer):  # This is just an example
    def __init__(self, params, *args, **kwargs):
        defaults = {'args': args, 'kwargs': kwargs}
        super().__init__(params, defaults)


class NonDiffMAML(Optimizer):  # copy pasted from torch.optim.SGD

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)


class MAML(DifferentiableOptimizer):  # copy pasted from DifferentiableSGD but with the g.detach() line of code

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if weight_decay != 0:
                    g = _add(g, weight_decay, p)
                if momentum != 0:
                    param_state = self.state[group_idx][p_idx]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = g
                    else:
                        buf = param_state['momentum_buffer']
                        buf = _add(buf.mul(momentum), 1 - dampening, g)
                        param_state['momentum_buffer'] = buf
                    if nesterov:
                        g = _add(g, momentum, buf)
                    else:
                        g = buf

                if self.fo:  # first-order
                    g = g.detach()  # dissallows flow of higher order grad while still letting params track gradients.
                group['params'][p_idx] = _add(p, -group['lr'], g)


higher.register_optim(NonDiffMAML, MAML)


class MAMLMetaLearner(nn.Module):
    def __init__(
            self,
            args,

            base_model,

            lr_inner=1e-1,  # careful with setting this to small or doing to few inner adaptation steps
            fo=False,
            inner_debug=False,
            target_type='classification'
    ):
        super().__init__()
        self.args = args  # args for experiment
        self.base_model = base_model

        self.lr_inner = lr_inner
        self.fo = fo
        self.inner_debug = inner_debug

        self.target_type = target_type

    def forward(self, spt_x, spt_y, qry_x, qry_y, training: bool = True, debug: bool = False):
        """Does L(A(theta,S), Q) = sum^N_{t=1} L(A(theta,S_t),Q_t) where A(theta,S) is the inner-adaptation loop.
        It also accumulates the gradient (for memory efficiency) for the outer-optimizer to later use

        Decision for BN/eval:
        - during training always use .train().
        During eval use the meta-train stats so do .eval() (and using .train() is always wrong since it cheats).
        Having track_running_stats=False seems overly complicated and nobody seems to use it...so why use it?

        ref for BN/eval:
            - https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
            - https://github.com/tristandeleu/pytorch-maml/issues/19

        Args:
            spt_x ([type]): x's for support set. Example shape [N,k_shot,D] D=1 or D=[C,H,W]
            spt_y ([type]): y's/target value for support set. Example shape [N,k_eval] or [N,k_eval,D]
            qry_x ([type]): x's for query set. Example shape [N,C,D] D=1 or D=[C,H,W]
            qry_y ([type]): y's/target value for query set. Example shape [N,k_eval] or [N,k_eval,D]

        Returns:
            [type]: [description]
        """
        print('USING NEW!') if debug else None
        # inner_opt = torch_uu.optim.SGD(self.base_model.parameters(), lr=self.lr_inner)
        inner_opt = NonDiffMAML(self.base_model.parameters(), lr=self.lr_inner)
        # print(f'{inner_opt=}')
        # inner_opt = torch_uu.optim.Adam(self.base_model.parameters(), lr=self.lr_inner)
        self.args.inner_opt_name = str(inner_opt)

        # self.base_model.train() if self.args.split == 'train' else self.base_model.eval()
        # - todo: warning, make sure this works if using in the future
        self.base_model.train() if training else self.base_model.eval()
        meta_batch_size = spt_x.size(0)
        meta_losses, meta_accs = [], []
        for t in range(meta_batch_size):
            spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
            # - Inner Loop Adaptation
            with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights,
                                      track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
                diffopt.fo = self.fo
                # print(f'>maml_old (before inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')
                for i_inner in range(self.args.nb_inner_train_steps):
                    # fmodel.train()  # omniglot doesn't have this here, it has a single one at the top https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py#L116

                    # base/child model forward pass
                    spt_logits_t = fmodel(spt_x_t)
                    inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
                    # inner_train_err = calc_error(mdl=fmodel, X=S_x, Y=S_y)  # for more advanced learners like meta-lstm

                    # inner-opt update
                    diffopt.step(inner_loss)
            # print(f'>>maml_old (after inner adapt): {fmodel.model.features.conv1.weight.norm(2)=}')

            # Evaluate on query set for current task
            qry_logits_t = fmodel(qry_x_t)
            qry_loss_t = self.args.criterion(qry_logits_t, qry_y_t)

            # Accumulate gradients wrt meta-params for each task: https://github.com/facebookresearch/higher/issues/104
            (qry_loss_t / meta_batch_size).backward()  # note this is more memory efficient (as it removes intermediate data that used to be needed since backward has already been called)

            # get accuracy
            if self.target_type == 'classification':
                qry_acc_t = calc_accuracy_from_logits(y_logits=qry_logits_t, y=qry_y_t)
            else:
                qry_acc_t = r2_score_from_torch(qry_y_t, qry_logits_t)
                # qry_acc_t = compressed_r2_score(y_true=qry_y_t.detach().numpy(), y_pred=qry_logits_t.detach().numpy())

            # collect losses & accs for logging/debugging
            meta_losses.append(qry_loss_t.item())
            meta_accs.append(qry_acc_t)

        assert len(meta_losses) == meta_batch_size
        meta_loss = np.mean(meta_losses)
        meta_acc = np.mean(meta_accs)
        meta_loss_std = np.std(meta_losses)
        meta_acc_std = np.std(meta_accs)
        return meta_loss, meta_acc, meta_loss_std, meta_acc_std
        # return meta_loss, meta_acc

    # def functional_similarities(self, spt_x, spt_y, qry_x, qry_y, layer_names, iter_tasks=None):
    #     """
    #     :return: sims = dictionary of metrics of tensors
    #     sims = {
    #         (cxa) cca, cka = tensor([I, L]),
    #         (l2) nes, cosine = tensor([I, L, K_eval])
    #         ned_output = tensor([I])
    #         }
    #     """
    #     # get difference
    #     inner_opt = NonDiffMAML(self.base_model.parameters(), lr=self.lr_inner)
    #
    #     meta_batch_size = spt_x.size(0)
    #     # print(f'meta_batch_size = {meta_batch_size}')
    #     iter_tasks = meta_batch_size if iter_tasks is None else iter_tasks
    #     # sims is an T by L lists of lists, i.e. each row corresponds to similarities for a specific task for all layers
    #     sims = {'cca': [], 'cka': [], 'nes': [], 'cosine': [], 'nes_output': [], 'query_loss': []}
    #     # compute similarities
    #     self.base_model.eval()
    #     for t in range(meta_batch_size):
    #         spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
    #         # Inner Loop Adaptation
    #         with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights,
    #                                   track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
    #             diffopt.fo = self.fo
    #             for i_inner in range(self.args.nb_inner_train_steps):
    #                 # print(f'{i_inner=}')
    #                 # base/child model forward pass
    #                 spt_logits_t = fmodel(spt_x_t)
    #                 inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
    #                 # inner-opt update
    #                 diffopt.step(inner_loss)
    #
    #             # get l2 sims per layer (T by L by k_eval)
    #             nes = self.get_l2_similarities_per_layer(self.base_model, fmodel, qry_x_t, layer_names, sim_type='nes_torch')
    #             cosine = self.get_l2_similarities_per_layer(self.base_model, fmodel, qry_x_t, layer_names, sim_type='cosine_torch')
    #             sims['nes'].append(nes)
    #             sims['cosine'].append(cosine)
    #
    #             # (T by 1)
    #             y = self.base_model(qry_x_t)
    #             y_adapt = fmodel(qry_x_t)
    #             # dim=0 because we have single numbers and we are taking the NES in the batch direction
    #             nes_output = uutils.torch.ned_torch(y.squeeze(), y_adapt.squeeze(), dim=0).item()
    #             sims['nes_output'].append(nes_output)
    #
    #             query_loss = self.args.criterion(y, y_adapt).item()
    #             sims['query_loss'].append(query_loss)
    #         if t + 1 >= iter_tasks:
    #             break
    #     # convert everything to torch_uu tensors
    #     # sims = {k: tensorify(v) for k, v in sims.items()}
    #     similarities = {}
    #     for k, v in sims.items():
    #         if k in ['cca', 'cka']:
    #             continue
    #         # print(f'{k}')
    #         similarities[k] = tensorify(v)
    #     return similarities

    def compute_functional_difference(self, spt_x, spt_y,
                                    qry_x, qry_y,
                                    layer_names,
                                    parallel: bool = False,
                                    iter_tasks=None):
        return self.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, layer_names, parallel, iter_tasks,
                                                    metric_as_dist=True)

    # def parallel_functional_similarities(self, spt_x, spt_y, qry_x, qry_y, layer_names, iter_tasks=None):
    #     return self.compute_functional_similarities(spt_x, spt_y, qry_x, qry_y, layer_names, parallel=True, iter_tasks=iter_tasks)

    def compute_functional_similarities(self, spt_x, spt_y,
                                        qry_x, qry_y,
                                        layer_names,
                                        parallel: bool = False,
                                        iter_tasks=None,
                                        metric_as_dist: bool = False
                                        ) -> dict[torch.Tensor]:
        """
        :return: sims = dictionary of metrics of tensors
        sims = {
            (cxa) cca, cka = tensor([I, L]),
            (l2) nes, cosine = tensor([I, L, K_eval])
            nes_output = tensor([I])
            }

        Important note:
        -When a Tensor is sent to another process, the Tensor data is shared. If torch.Tensor.grad is not None,
            it is also shared.
        - After a Tensor without a torch.Tensor.grad field is sent to the other process,
            it creates a standard process-specific .grad Tensor that is not automatically shared across all processes,
            unlike how the Tensor’s data has been shared.
            - this is good! that way when different tasks are being adapted with MAML, their gradients don't "crash"
        - above from docs on note box: https://pytorch.org/docs/stable/notes/multiprocessing.html

        Note:
            - you probably do not have to call share_memory() but it's probably safe to do so. Since docs say the following:
                Moves the underlying storage to shared memory. This is a no-op if the underlying storage is already in
                shared memory and for CUDA tensors. Tensors in shared memory cannot be resized.
                https://pytorch.org/docs/stable/tensors.html#torch.Tensor.share_memory_

        To increase file descriptors
            ulimit -Sn unlimited
        """
        meta_batch_size = spt_x.size(0)
        iter_tasks = meta_batch_size if iter_tasks is None else min(iter_tasks, meta_batch_size)
        T: int = min(iter_tasks, meta_batch_size)

        # - prepare some args
        # to have shared memory tensors
        self.base_model.share_memory()
        # this is needed so that each process has access to the layer names needed to compute sims, a bit ugly oh well.
        self.layer_names = layer_names
        inner_opt = NonDiffMAML(self.base_model.parameters(), lr=self.lr_inner)
        self.inner_opt = inner_opt

        # - args for to compute similarities for all tasks
        spt_x.share_memory_(), spt_y.share_memory_(), qry_x.share_memory_(), qry_y.share_memory_()
        batch_of_tasks: Batch[Task] = [(spt_x[t], spt_y[t], qry_x[t], qry_y[t]) for t in range(iter_tasks)]

        # -- Compute similarities for all tasks in meta-batch
        # num_procs = iter_tasks
        if parallel:
            torch.multiprocessing.set_sharing_strategy('file_system')
            num_procs = 8
            # num_procs = torch.multiprocessing.cpu_count() - 2
            with Pool(num_procs) as pool:
                # note: not a problem that it returns it in undosrted since the model's d is evaluated per task and the order of tasks doesn't matter. The order of layers will NOT be affected :)
                sims_all_tasks: list[dict] = pool.map(self.compute_sim_for_current_task, batch_of_tasks)
        else:
            # sims for each task, each dict has the sims for each metric
            sims_all_tasks: list[dict] = []  # len(sims_all_tasks) = T
            for t in range(T):
                # sims for current task for each type of metric
                sims_for_current_task: dict = self.compute_sim_for_current_task(batch_of_tasks[t])
                sims_all_tasks.append(sims_for_current_task)
        assert T == len(sims_all_tasks), f'We should have one sim for each task but we have {T} tasks and have {len(sims_all_tasks)} sims, mistmatch of tasks expected and used to compute sims.'

        # -- from [Tasks, Sims] -> [Sims, Tasks]
        sims = {'cca': [], 'cka': [], 'op': [],  # [T, L]
                'nes': [], 'cosine': [],  # [T, L, K]
                'nes_output': [], 'query_loss': []  # [T, 1]
                }
        for t in range(T):
            sims_all_metrics_current_task: dict = sims_all_tasks[t]  # sims for each each metric for 1 task
            for metric, s in sims_all_metrics_current_task.items():
                # [1, L] for cca, cka, op
                # [1, L, K] for nes, cosine
                # [1, 1] for nes_output, query_loss
                ss = 1.0 - s if metric_as_dist and metric != 'query_loss' else s
                # print(f'd={1.0 - s}')
                if metric != 'cosine':
                    error_tolerance: float = -0.0001
                    assert (ss >= error_tolerance).all(), f'Distances are positive but got a negative value somewhere for metric {metric=}.'
                sims[metric].append(ss)

        # convert everything to torch.tensors
        similarities = {}
        for metric, v in sims.items():
            tensor_sims: torch.Tensor = tensorify(v)
            similarities[metric] = tensor_sims
        return similarities

    def compute_sim_for_current_task(self, task: tuple[Spt_x, Spt_y, Qry_x, Qry_y]) -> dict:
        """
        Computes the similarity s(f, A(f)) for a single task represented as the data for it in the support & query sets.

        Returns a dict of metrics for each task e.g. [1, L] for CCA
        NOTE: T = 1
        'cca': cca,  # [T, L]
        'cka': cka,  # [T, L]
        'op': op,  # [T, L]
        'nes': nes,  # [T, L, k_eval]
        'cosine': cosine,  # [T, L, k_eval]
        'nes_output': nes_output,  # [T, 1], a single value since it's a metric for the output of the network
        'query_loss': query_loss   # [T, 1], since it's a single loss value
        """
        # unpack args
        T, L = 1, len(self.layer_names)
        spt_x_t, spt_y_t, qry_x_t, qry_y_t = task
        # compute sims
        self.base_model.eval()
        with higher.innerloop_ctx(self.base_model, self.tesi, copy_initial_weights=self.args.copy_initial_weights,
                                  track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
            diffopt.fo = self.fo
            for i_inner in range(self.args.nb_inner_train_steps):
                # base/child model forward pass
                spt_logits_t = fmodel(spt_x_t)
                inner_loss = self.args.criterion(spt_logits_t, spt_y_t)
                # inner-opt update
                diffopt.step(inner_loss)

            # get similarities cca, cka, nes, cosine and nes_output
            # get CCA & CKA per layer
            x = qry_x_t
            if torch.cuda.is_available():
                x = x.cuda()
            # [T, L], note: T=1 since this function computes for a single task
            cca: list[float] = self.get_cxa_similarities_per_layer(self.base_model, fmodel, x, self.layer_names,
                                                      sim_type='pwcca')
            cka: list[float] = self.get_cxa_similarities_per_layer(self.base_model, fmodel, x, self.layer_names,
                                                      sim_type='lincka')
            assert T == 1
            assert len(cca) == L
            # -- get l2 sims per layer
            # [T, L], note: T=1 since this function computes for a single task
            op = self.get_l2_similarities_per_layer(self.base_model, fmodel, x, self.layer_names,
                                                     sim_type='op_torch')
            # [T, L, k_eval], note: T=1 since this function computes for a single task
            nes = self.get_l2_similarities_per_layer(self.base_model, fmodel, x, self.layer_names,
                                                     sim_type='nes_torch')
            cosine = self.get_l2_similarities_per_layer(self.base_model, fmodel, x, self.layer_names,
                                                        sim_type='cosine_torch')
            assert T == 1, f'Expected {T} to be 1.'
            assert len(nes) == L, f'Expected {len(nes)=} to be equal to num_layer=L={L}'
            assert len(nes[0]) == self.args.k_eval, f'Error, not the same k_eval size, got {len(nes[0])} ' \
                                                    f'but expected {self.args.k_eval}'

            # [T, 1], L is not present since it's for the output, note: T=1 since this function computes for a single task
            y = self.base_model(qry_x_t)
            y_adapt = fmodel(qry_x_t)
            # dim=0 because we have single numbers and we are taking the NES in the batch direction
            nes_output = uutils.torch_uu.nes_torch(y.squeeze(), y_adapt.squeeze(), dim=0).item()
            assert T == 1
            assert isinstance(nes_output, float)

            query_loss = self.args.criterion(y, y_adapt).item()
        # sims = [cca, cka, nes, cosine, nes_output, query_loss]
        sims = {'cca': cca, 'cka': cka, 'op': op, 'nes': nes, 'cosine': cosine, 'nes_output': nes_output, 'query_loss': query_loss}
        sims = {metric: tensorify(sim).detach() for metric, sim in sims.items()}
        return sims

    def get_cxa_similarities_per_layer(self, mdl1, mdl2, X, layer_names, sim_type='pwcca'):
        # get [..., s_l, ...] cca sim per layer (for this data set)
        from uutils.torch_uu import cxa_sim

        sims_per_layer = []
        for layer_name in layer_names:
            # sim = cxa_sim(mdl1, mdl2, X, layer_name, cca_size=self.args.cca_size, iters=1, cxa_sim_type=sim_type)
            sim = cxa_sim(mdl1, mdl2, X, layer_name, iters=1, cxa_sim_type=sim_type)
            sims_per_layer.append(sim)
        return sims_per_layer  # [..., s_l, ...]_l

    def get_l2_similarities_per_layer(self, mdl1, mdl2, X, layer_names, sim_type='nes_torch'):
        import copy
        from uutils.torch_uu import l2_sim_torch
        # get [..., s_l, ...] sim per layer (for this data set)
        modules = zip(mdl1.named_children(), mdl2.named_children())
        sims_per_layer = []
        out1 = X
        out2 = X
        for (name1, m1), (name2, m2) in modules:
            # print(f'{(name1, m1), (name2, m2)}')
            # - always do the forward pass of the net for all layers
            out1 = m1(out1)
            m2_callable = copy.deepcopy(m1)
            m2_callable.load_state_dict(m2.state_dict())
            out2 = m2_callable(out2)
            # - only collect values for chosen layers
            if name1 in layer_names:
                # out1, out2 = normalize_matrix_for_similarity(out1, dim=1), normalize_matrix_for_similarity(out2, dim=1)
                sim = l2_sim_torch(out1, out2, sim_type=sim_type)
                sims_per_layer.append(sim)
        return sims_per_layer  # [[s_k,l]_k]_l = [..., [...,s_k,l, ...]_k, ...]_l

    def eval(self):
        """
        Note: decision is to do .train() for all meta-train and .eval() for meta-eval.

        ref: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-du
        """
        self.base_model.eval()

    def eval_no_running_statistics(self):
        """ ref: https://stats.stackexchange.com/questions/544048/what-does-the-batch-norm-layer-for-maml-model-agnostic-meta-learning-do-for-d """
        self.base_model.eval()
        assert False, 'Not implemented'
        # we would do something like set_running_stats_to_false(self.base_model) here

    def parameters(self):
        return self.base_model.parameters()

    def regression(self):
        self.target_type = 'regression'
        self.args.target_type = 'regression'

    def classification(self):
        self.target_type = 'classification'
        self.args.target_type = 'classification'

    def cuda(self):
        self.base_model.cuda()

# Tests

class Flatten(nn.Module):
    def forward(self, input):  # (k, C, H, W)
        k_examples = input.size(0)
        out = input.view(k_examples, -1)
        return out  # (K, C*H*W)


def maml_directly_good_accumulator_simple_test(verbose=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    from collections import OrderedDict

    from types import SimpleNamespace

    # training config
    args = SimpleNamespace()
    # args.device = torch_uu.device("cuda" if torch_uu.cuda.is_available() else "cpu")

    args.split = 'meta-train'
    args.track_higher_grads = True  # if True, during unrolled optimization the graph be retained, and the fast weights will bear grad funcs, so as to permit backpropagation through the optimization process. False during test time for efficiency reasons
    args.copy_initial_weights = False  # if False then we train the base models initial weights (i.e. the base model's initialization)
    args.episodes = 7
    args.nb_inner_train_steps = 5
    args.outer_lr = 1e-2
    args.lr_inner = 1e-1  # carefuly if you don't know how to change this one
    args.fo = True
    # N-way, K-shot, with k_eval points
    args.meta_batch_size = 50
    args.k_shot, args.k_eval = 5, 15
    args.n_classes = 5
    D = 1
    # loss for tasks
    args.criterion = nn.CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
    # get base model
    nb_hidden_units = 1
    base_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Linear(D, nb_hidden_units)),
        ('act', nn.LeakyReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear(nb_hidden_units, args.n_classes))
    ]))
    meta_learner = MAMLMetaLearner(args, base_model, args.lr_inner, args.fo)
    meta_params = meta_learner.parameters()
    outer_opt = optim.Adam(meta_params, lr=args.outer_lr)
    for episode in range(args.episodes):
        ## get fake support & query data from batch of N tasks
        spt_x, qry_x = torch.randn(args.meta_batch_size, args.n_classes * args.k_shot, D), torch.randn(
            args.meta_batch_size, args.n_classes * args.k_eval, D)
        spt_y, qry_y = torch.randint(0, args.n_classes,
                                     [args.meta_batch_size, args.n_classes * args.k_shot]), torch.randint(0,
                                                                                                          args.n_classes,
                                                                                                          [
                                                                                                              args.meta_batch_size,
                                                                                                              args.n_classes * args.k_eval])
        ## Compute grad Meta-Loss
        meta_loss, meta_acc = meta_learner(spt_x, spt_y, qry_x, qry_y)
        if verbose:
            pass
            # no need of this, None is already checked
            # print(f'base_model.conv1.weight.grad= {base_model.conv1.weight.grad}')
            # print(f'base_model.conv1.bias.grad = {base_model.conv1.bias.grad}')
            # print(f'base_model.fc1.weight.grad = {base_model.fc1.weight.grad}')
            # print(f'base_model.fc1.bias.grad = {base_model.fc1.bias.grad}')
        assert (base_model.conv1.weight.grad is not None)
        assert (base_model.fc1.weight.grad is not None)
        assert (meta_loss != 0)
        outer_opt.step()
        outer_opt.zero_grad()


if __name__ == "__main__":
    maml_directly_good_accumulator_simple_test(verbose=True)
    print('Done, all Tests Passed! \a')
