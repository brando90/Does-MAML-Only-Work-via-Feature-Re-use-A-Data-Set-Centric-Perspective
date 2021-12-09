import torch
import torch.nn as nn

import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from pathlib import Path

from types import SimpleNamespace

from sklearn.metrics import log_loss
from torch import Tensor

from uutils.logger import Logger

# import automl.child_models.learner_from_opt_as_few_shot_paper.Learner

# https://github.com/WangYueFt/rfs/blob/master/eval/meta_eval.py
from uutils.torch_uu import r2_score_from_torch
from uutils.torch_uu.models import getattr_model


class FitFinalLayer(nn.Module):

    def __init__(self,
                 args,
                 base_model,
                 target_type='classification',
                 classifier='LR'):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.target_type = target_type
        self.classifier = classifier

    def forward(self, spt_x, spt_y, qry_x, qry_y, training: bool = True):
        # Accumulate gradient of meta-loss wrt fmodel.param(t=0)
        meta_batch_size = spt_x.size(0)
        meta_losses, meta_accs = [], []
        for t in range(meta_batch_size):
            spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]

            self.base_model.train() if training else self.base_model.eval()
            spt_embeddings_t = self.get_embedding(spt_x_t, self.base_model).detach()
            qry_embeddings_t = self.get_embedding(qry_x_t, self.base_model).detach()
            if self.target_type == 'classification':
                spt_embeddings_t = spt_embeddings_t.view(spt_embeddings_t.size(0), -1).cpu().numpy()
                qry_embeddings_t = qry_embeddings_t.view(qry_embeddings_t.size(0), -1).cpu().numpy()
                spt_y_t = spt_y_t.view(-1).cpu().numpy()
                qry_y_t = qry_y_t.view(-1).cpu().numpy()

                # Inner-Adapt final layer with spt set
                mdl = LogisticRegression(random_state=0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                mdl.fit(spt_embeddings_t, spt_y_t)

                # Predict using adapted final layer with qrt set
                if self.classifier == 'LR':
                    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(spt_embeddings_t, spt_y_t)
                    query_y_pred_t = clf.predict(qry_embeddings_t)
                    query_y_probs_t = clf.predict_proba(qry_embeddings_t)
                elif self.classifier == 'NN':
                    query_y_pred_t = NN(spt_embeddings_t, spt_y_t, qry_embeddings_t)
                    raise ValueError(f'Not tested {self.classifier}')
                elif self.classifier == 'Cosine':
                    query_y_pred_t = Cosine(spt_embeddings_t, spt_y_t, qry_embeddings_t)
                    raise ValueError(f'Not tested {self.classifier}')
                # acc
                qry_loss_t = log_loss(qry_y_t, query_y_probs_t)
                qry_acc_t = metrics.accuracy_score(query_y_pred_t, qry_y_t)
            elif self.target_type == 'regression':
                # Inner-Adapt final adapted layer with spt set
                mdl = LinearRegression().fit(spt_embeddings_t, spt_y_t)
                # Predict using adapted final layer with qrt set
                query_y_pred_t = torch.Tensor(mdl.predict(qry_embeddings_t))
                qry_loss_t = self.args.criterion(query_y_pred_t, qry_y_t).item()
                qry_acc_t = r2_score_from_torch(qry_y_t, query_y_pred_t)
            else:
                raise ValueError(f'Not implement: {self.target_type}')

            # collect losses & accs for logging/debugging
            meta_losses.append(qry_loss_t)
            meta_accs.append(qry_acc_t)

        # Get average meta-loss/acc of the meta-learner i.e. 1/B sum_b Loss(qrt_i, f) = E_B E_K[loss(qrt[b,k], f)]
        # average loss on task of size K-eval over a meta-batch of size B (so B tasks)
        assert (len(meta_losses) == meta_batch_size)
        meta_loss = np.mean(meta_losses)
        meta_acc = np.mean(meta_accs)
        meta_loss_std = np.std(meta_losses)
        meta_acc_std = np.std(meta_accs)
        return meta_loss, meta_acc, meta_loss_std, meta_acc_std
        # return meta_loss, meta_accs

    def get_embedding(self, x: Tensor, base_model: nn.Module) -> Tensor:
        return get_embedding(x=x, base_model=base_model)

    def regression(self):
        self.target_type = 'regression'

    def classification(self):
        self.target_type = 'classification'

    def train(self):
        self.base_model.train()

    def eval(self):
        self.base_model.eval()


def get_adapted_according_to_ffl(base_model, spt_x_t, spt_y_t, qry_x_t, qry_y_t,
                                 layer_to_replace: str,
                                 training: bool = True,
                                 target_type: str = 'classification',
                                 classifier: str = 'LR', ) -> nn.Module:
    """
    Return the adapted model such that the final layer has the LR fined tuned model.
    """
    # spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t], spt_y[t], qry_x[t], qry_y[t]
    base_model.train() if training else base_model.eval()
    spt_embeddings_t = get_embedding(spt_x_t, base_model).detach()
    qry_embeddings_t = get_embedding(qry_x_t, base_model).detach()
    if target_type == 'classification':
        spt_embeddings_t = spt_embeddings_t.view(spt_embeddings_t.size(0), -1).cpu().numpy()
        qry_embeddings_t = qry_embeddings_t.view(qry_embeddings_t.size(0), -1).cpu().numpy()
        spt_y_t = spt_y_t.view(-1).cpu().numpy()
        qry_y_t = qry_y_t.view(-1).cpu().numpy()

        # Inner-Adapt final layer with spt set
        mdl = LogisticRegression(random_state=0,
                                 solver='lbfgs',
                                 max_iter=1000,
                                 multi_class='multinomial')
        mdl.fit(spt_embeddings_t, spt_y_t)

        # Predict using adapted final layer with qrt set
        if classifier == 'LR':
            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                     multi_class='multinomial')
            clf.fit(spt_embeddings_t, spt_y_t)
            # query_y_pred_t = clf.predict(qry_embeddings_t)
            query_y_probs_t = clf.predict_proba(qry_embeddings_t)

            # - get layer_to_replace e.g. model.cls
            module: nn.Module = getattr_model(base_model, layer_to_replace)
            assert module is not None, f'Final layer module is None instead of a pytorch module see: {module=}'
            # assert module is base_model.model.cls

            # - replace weights into model
            # coef_ndarray of shape (1, n_features) or (n_classes, n_features)
            new_weights: np.ndarray = clf.coef_  # (n_classes, n_features) -> [n_features, n_classes]
            new_biases: np.ndarray = clf.intercept_  # (n_classes,) -> [n_features,]
            num_classes, num_features = new_weights.shape[0], new_weights.shape[1]  # in_features, out_features=Dout
            module.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(torch.float32))
            module.bias = torch.nn.Parameter(torch.from_numpy(new_biases).to(torch.float32))
            assert module.weight.size() == torch.Size([num_classes, num_features]), f'Error not the same: ' \
                                                                                    f'{module.weight.size()}, {torch.Size([num_features, num_classes])}'
            assert module.bias.size() == torch.Size([num_classes])
            # query_y_probs_t = clf.predict_proba(qry_embeddings_t[:2, :])
            # out_mdl = torch.softmax(mdl(torch.from_numpy(qry_embeddings_t[2:, :]), dim=1))
            # assert np.isclose(out_mdl.detach().cpu().numpy(), query_y_probs_t).all()
        elif classifier == 'NN':
            query_y_pred_t = NN(spt_embeddings_t, spt_y_t, qry_embeddings_t)
            raise ValueError(f'Not tested {classifier}')
        elif classifier == 'Cosine':
            query_y_pred_t = Cosine(spt_embeddings_t, spt_y_t, qry_embeddings_t)
            raise ValueError(f'Not tested {classifier}')
        # acc
        # qry_loss_t = log_loss(qry_y_t, query_y_probs_t)
        # qry_acc_t = metrics.accuracy_score(query_y_pred_t, qry_y_t)
    elif target_type == 'regression':
        # Inner-Adapt final adapted layer with spt set
        # mdl = LinearRegression().fit(spt_embeddings_t, spt_y_t)
        # # Predict using adapted final layer with qrt set
        # query_y_pred_t = torch.Tensor(mdl.predict(qry_embeddings_t))
        # qry_loss_t = args.criterion(query_y_pred_t, qry_y_t).item()
        # qry_acc_t = r2_score_from_torch(qry_y_t, query_y_pred_t)
        assert False
    else:
        raise ValueError(f'Not implement: {target_type}')
    return base_model

def get_embedding(x: Tensor, base_model: nn.Module) -> Tensor:
    """ apply f until the last layer, instead return that as the embedding """
    out = x
    # if it has a get embedding later
    if hasattr(base_model, 'get_embedding'):
        out = base_model.get_embedding(x)
        return out
    # for handling models with self.model.features self.model.cls format
    if hasattr(base_model, 'model'):
        out = base_model.model.features(x)
        return out
    # for handling synthetic base models
    # https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/3
    for name, m in base_model.named_children():
        if 'final' in name:
            return out
        if 'l2' in name and 'final' not in name:  # cuz I forgot to write final...sorry!
            return out
        if name == 'fc':
            return out
        out = m(out)
    raise ValueError(
        'Your model does not have an explicit point where the embedding starts (i.e. the word final) or other bug in model')

def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred


# - tests

def setup_and_get_logger(args):
    args.logging = True
    args.log_root = Path('//experiments/logs/').expanduser()
    current_logs_dir = Path(f'logs')
    args.current_logs_path = args.log_root / current_logs_dir
    args.current_logs_path.mkdir(parents=True, exist_ok=True)
    # set up path + log filename
    my_stdout_filename = Path('my_stdout.log')
    args.my_stdout_filepath = args.current_logs_path / my_stdout_filename
    # make logger
    logger = Logger(args)  # logs to file & console
    return logger


def f_rand_is_worse_than_f_avg_test():
    """
    check that f_avg > f_rand (is better) in meta-test
    """
    import copy

    from meta_learning.training.meta_training import meta_eval
    from meta_learning.datasets.rand_fnn import RandFNN

    from torchmeta.utils.data import BatchMetaDataLoader
    from torchmeta.transforms import ClassSplitter

    # experiment args
    args = SimpleNamespace()
    args.k_shots = 5
    args.k_eval = 5
    args.meta_batch_size_eval = 16
    args.num_workers = 4
    args.split = 'val'
    args.data_path = Path('~/data/LS/fully_connected_NN_mu1_0.05_std1_0.1_mu2_0.05_std2_0.025/').expanduser()
    args.criterion = nn.MSELoss()
    args.epochs_eval = 2
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = setup_and_get_logger(args)
    args.debug_test = False

    # get meta-sets dataloaders
    dataset_val = RandFNN(args.data_path, 'val')
    metaset_val = ClassSplitter(dataset_val, num_train_per_class=args.k_shots,
                                num_test_per_class=args.k_eval,
                                shuffle=True)
    meta_val_dataloader = BatchMetaDataLoader(metaset_val,
                                              batch_size=args.meta_batch_size_eval,
                                              num_workers=args.num_workers)
    # initial model
    db = torch.load(args.data_path / args.split / 'f_avg')
    f_avg = db['f_avg']
    f_rand = copy.deepcopy(f_avg)
    [layer.reset_parameters() for layer in f_rand.children() if hasattr(layer, 'reset_parameters')]

    # get meta learner
    meta_learner_f_avg = FitFinalLayer(args, f_avg)
    meta_learner_f_avg.regression()
    meta_learner_f_rand = FitFinalLayer(args, f_rand)
    meta_learner_f_rand.regression()

    acc_mean_f_avg, acc_std, loss_mean_f_avg, loss_std = meta_eval(args, meta_learner_f_avg, meta_val_dataloader)
    acc_mean_f_rand, acc_std, loss_mean_f_rand, loss_std = meta_eval(args, meta_learner_f_rand, meta_val_dataloader)

    print(f'loss_mean_f_avg = {loss_mean_f_avg}')
    print(f'loss_mean_f_avg = {loss_mean_f_rand}')
    assert (loss_mean_f_avg < loss_mean_f_rand)  # f_avg error better than f_rand


if __name__ == '__main__':
    print('__main__ started!')
    import time

    start = time.time()
    # test_f_rand_is_worse_than_f_avg()
    seconds = time.time() - start
    print(f'seconds = {seconds}, hours = {seconds / 60} \n\a')
