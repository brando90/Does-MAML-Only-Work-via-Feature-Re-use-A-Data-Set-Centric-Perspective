import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input): # (k, C, H, W)
        k_examples = input.size(0)
        out = input.view(k_examples, -1)
        return out # (K, C*H*W)

def test_maml_meta_learner(verbose=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import OrderedDict

    from types import SimpleNamespace

    ## training config
    args = SimpleNamespace()
    #args.device = torch_uu.device("cuda" if torch_uu.cuda.is_available() else "cpu")
    args.mode = 'meta-train'
    args.track_higher_grads = True # if True, during unrolled optimization the graph be retained, and the fast weights will bear grad funcs, so as to permit backpropagation through the optimization process. False during test time for efficiency reasons
    args.copy_initial_weights = False # if False then we train the base models initial weights (i.e. the base model's initialization)
    args.episodes = 5
    args.nb_inner_train_steps = 5
    args.outer_lr = 1e-2
    args.inner_lr = 1e-1 # carefuly if you don't know how to change this one
    # N-way, K-shot, with k_eval points
    args.k_shot, args.k_eval = 5, 15
    args.n_classes = 5 
    C,H,W = [3, 84, 84] # like mini-imagenet
    # loss for tasks
    args.criterion = nn.CrossEntropyLoss() # The input is expected to contain raw, unnormalized scores for each class.
    ## get base model
    kernel_size = 3
    base_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=3,out_channels=7,kernel_size=kernel_size)), # H' = H - k + 1
        ('act', nn.LeakyReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear( (H-kernel_size+1)*(W-kernel_size+1)*7, args.n_classes))
        ]))
    ## maml-meta-learner
    maml = MAMLMetaLearner(args, base_model)
    ## get outer optimizer (not differentiable nor trainable)
    outer_opt = optim.Adam([{'params': base_model.parameters()}], lr=args.outer_lr)
    for episode in range(args.episodes):
        ## get fake support & query data from batch of N tasks
        spt_x, spt_y = torch.randn(args.n_classes, args.k_shot, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_shot) for label in range(args.n_classes) ])
        qry_x, qry_y = torch.randn(args.n_classes, args.k_eval, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_eval) for label in range(args.n_classes) ])
        ## Acculumate gradients of meta-loss
        losess, meta_accs = maml(spt_x, spt_y, qry_x, qry_y)
        ## outer update
        if verbose:
            print(f'--> episode = {episode}')
            print(f'meta-loss = {sum(losess)}, meta-accs = {sum(meta_accs)}')
            print(f'base_model.conv1.weight.grad= {base_model.conv1.weight.grad}')
            print(f'base_model.conv1.bias.grad = {base_model.conv1.bias.grad}')
            print(f'base_model.fc1.weight.grad = {base_model.fc1.weight.grad}')
            print(f'base_model.fc1.bias.grad = {base_model.fc1.bias.grad}')
        assert(base_model.conv1.weight.grad is not None)
        assert(base_model.fc1.weight.grad is not None)
        outer_opt.step()
        outer_opt.zero_grad()

def test_maml_directly_bad_accumulator(verbose = False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import OrderedDict

    from types import SimpleNamespace

    ## training config
    args = SimpleNamespace()
    #args.device = torch_uu.device("cuda" if torch_uu.cuda.is_available() else "cpu")
    args.mode = 'meta-train'
    args.track_higher_grads = True # if True, during unrolled optimization the graph be retained, and the fast weights will bear grad funcs, so as to permit backpropagation through the optimization process. False during test time for efficiency reasons
    args.copy_initial_weights = False # if False then we train the base models initial weights (i.e. the base model's initialization)
    args.episodes = 5
    args.nb_inner_train_steps = 5
    args.outer_lr = 1e-2
    args.inner_lr = 1e-1 # carefuly if you don't know how to change this one
    # N-way, K-shot, with k_eval points
    args.k_shot, args.k_eval = 5, 15
    args.n_classes = 5 
    C,H,W = [3, 84, 84] # like mini-imagenet
    # loss for tasks
    args.criterion = nn.CrossEntropyLoss() # The input is expected to contain raw, unnormalized scores for each class.
    ## get base model
    kernel_size = 3
    base_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=3,out_channels=7,kernel_size=kernel_size)), # H' = H - k + 1
        ('act', nn.LeakyReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear( (H-kernel_size+1)*(W-kernel_size+1)*7, args.n_classes))
        ]))
    ## maml-meta-learner
    #maml = MAMLMetaLearner(args, base_model)
    ## get outer optimizer (not differentiable nor trainable)
    outer_opt = optim.Adam([{'params': base_model.parameters()}], lr=args.outer_lr)
    for episode in range(args.episodes):
        ## get fake support & query data from batch of N tasks
        spt_x, spt_y = torch.randn(args.n_classes, args.k_shot, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_shot) for label in range(args.n_classes) ])
        qry_x, qry_y = torch.randn(args.n_classes, args.k_eval, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_eval) for label in range(args.n_classes) ])
        ## Compute sum^N_{t=1} L(A(theta,S_t),Q_t)
        ## get differentiable & trainable (parametrized) inner optimizer
        inner_opt = torch.optim.SGD(base_model.parameters(), lr=args.inner_lr)
        ## Compute grad_{\theta^<0,T>} sum^N_{t=1} L(A(theta,S_t),Q_t)
        nb_tasks = spt_x.size(0) # extract N tasks. Note M=N
        meta_losses, meta_accs = [], []
        # computes 1/M \sum^M_t L(A(\theta,S_t), Q_t)
        meta_loss = 0
        for t in range(nb_tasks):
            with higher.innerloop_ctx(base_model, inner_opt, copy_initial_weights=args.copy_initial_weights, track_higher_grads=args.track_higher_grads) as (fmodel, diffopt):
                spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t,:,:,:], spt_y[t,:], qry_x[t,:,:,:], qry_y[t,:]
                ## Inner-Adaptation Loop for the current task: \theta^<,t_Outer,i_inner+1> := \theta^<t_Outer,T> - lr_inner * \grad _{\theta^{<t_Outer,t_inner>} L(\theta^{<t_Outer,t_inner>},S_t)
                # since S_t is so small k_shot (1 or 5, for each class/task) we use the whole thing
                for i_inner in range(args.nb_inner_train_steps):
                    fmodel.train()
                    # base/child model forward pass
                    S_logits_t = fmodel(spt_x_t) 
                    inner_loss = args.criterion(S_logits_t, spt_y_t)
                    # inner-opt update
                    diffopt.step(inner_loss)
                ## Evaluate on query set for current task
                qry_logits_t = fmodel(qry_x_t)
                qry_loss_t = args.criterion(qry_logits_t,  qry_y_t)
                ## Update the model's meta-parameters to optimize the query losses across all of the tasks sampled in this batch. This unrolls through the gradient steps.
                meta_loss += qry_loss_t
                #qry_loss_t.backward() # this accumualtes the gradients in a memory efficient way to compute the desired gradients on the meta-loss for each task
            meta_loss = meta_loss / nb_tasks
        ## outer update
        meta_loss.backward()
        if verbose:
            print(f'--> episode = {episode}')
            print(f'meta-loss = {sum(meta_losses)}, meta-accs = {sum(meta_accs)}')
            print(f'base_model.conv1.weight.grad= {base_model.conv1.weight.grad}')
            print(f'base_model.conv1.bias.grad = {base_model.conv1.bias.grad}')
            print(f'base_model.fc1.weight.grad = {base_model.fc1.weight.grad}')
            print(f'base_model.fc1.bias.grad = {base_model.fc1.bias.grad}')
        assert( base_model.conv1.weight.grad is not None )
        assert( base_model.fc1.weight.grad is not None )
        outer_opt.step()
        outer_opt.zero_grad()

def test_higher_maml_directly_good_accumulator(verbose = False):
    print('------> test_higher_maml_directly_good_accumulator')
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import higher
    import torch.nn.functional as F

    from uutils.torch_uu import calc_accuracy

    from collections import OrderedDict

    from types import SimpleNamespace

    ## training config
    args = SimpleNamespace()
    #args.device = torch_uu.device("cuda" if torch_uu.cuda.is_available() else "cpu")
    args.mode = 'meta-train'
    args.track_higher_grads = True # if True, during unrolled optimization the graph be retained, and the fast weights will bear grad funcs, so as to permit backpropagation through the optimization process. False during test time for efficiency reasons
    args.copy_initial_weights = False # if False then we train the base models initial weights (i.e. the base model's initialization)
    args.episodes = 1
    args.nb_inner_train_steps = 1
    args.outer_lr = 1e-2
    args.inner_lr = 1e-1 # carefuly if you don't know how to change this one
    # N-way, K-shot, with k_eval points
    args.k_shot, args.k_eval = 1, 15
    args.n_classes = 2 
    C,H,W = [3, 84, 84] # like mini-imagenet
    # loss for tasks
    args.criterion = nn.CrossEntropyLoss() # The input is expected to contain raw, unnormalized scores for each class.
    args.criterion = F.cross_entropy
    ## get base model
    kernel_size = 3
    hidden_units = 1
    base_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=C,out_channels=hidden_units,kernel_size=kernel_size)), # H' = H - k + 1
        ('act', nn.ReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear( (H-kernel_size+1)*(W-kernel_size+1)*hidden_units, args.n_classes))
        ]))
    ## maml-meta-learner
    #maml = MAMLMetaLearner(args, base_model)
    ## get outer optimizer (not differentiable nor trainable)
    outer_opt = optim.Adam([{'params': base_model.parameters()}], lr=args.outer_lr)
    for episode in range(args.episodes):
        ## get fake support & query data from batch of N tasks
        spt_x, spt_y = torch.randn(args.n_classes, args.k_shot, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_shot) for label in range(args.n_classes) ])
        qry_x, qry_y = torch.randn(args.n_classes, args.k_eval, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_eval) for label in range(args.n_classes) ])
        ## Compute sum^N_{t=1} L(A(theta,S_t),Q_t)
        ## get differentiable & trainable (parametrized) inner optimizer
        inner_opt = torch.optim.SGD(base_model.parameters(), lr=args.inner_lr)
        ## Compute grad_{\theta^<0,T>} sum^N_{t=1} L(A(theta,S_t),Q_t)
        nb_tasks = spt_x.size(0) # extract N tasks. Note M=N
        meta_losses, meta_accs = [], []
        # computes 1/M \sum^M_t L(A(\theta,S_t), Q_t)
        meta_loss = 0
        for t in range(nb_tasks):
            print(f'task = t = {t}')
            spt_x_t, spt_y_t, qry_x_t, qry_y_t = spt_x[t,:,:,:], spt_y[t,:], qry_x[t,:,:,:], qry_y[t,:]
            #print(f'spt_x_t, spt_y_t = {spt_x_t, spt_y_t}')
            #print(f'qry_x_t, qry_y_t = {qry_x_t, qry_y_t}')
            with higher.innerloop_ctx(base_model, inner_opt, copy_initial_weights=args.copy_initial_weights, track_higher_grads=args.track_higher_grads) as (fmodel, diffopt):
                ## Inner-Adaptation Loop for the current task: \theta^<,t_Outer,i_inner+1> := \theta^<t_Outer,T> - lr_inner * \grad _{\theta^{<t_Outer,t_inner>} L(\theta^{<t_Outer,t_inner>},S_t)
                # since S_t is so small k_shot (1 or 5, for each class/task) we use the whole thing
                print(f'BEFORE sum(base_model.norm) = {sum([p.norm(2) for p in fmodel.parameters()])}')
                for i_inner in range(args.nb_inner_train_steps):
                    fmodel.train()
                    # base/child model forward pass
                    S_logits_t = fmodel(spt_x_t) 
                    inner_loss = args.criterion(S_logits_t, spt_y_t)
                    # inner-opt update
                    diffopt.step(inner_loss)
                print(f'AFTER sum(base_model.norm) = {sum([p.norm(2) for p in fmodel.parameters()])}')
                ## Evaluate on query set for current task
                qry_logits_t = fmodel(qry_x_t)
                print(f'qry_logits_t = {qry_logits_t}')
                print(f'qry_y_t = {qry_y_t}')
                qry_loss_t = args.criterion(qry_logits_t,  qry_y_t)
                ## Update the model's meta-parameters to optimize the query losses across all of the tasks sampled in this batch. This unrolls through the gradient steps.
                meta_loss += qry_loss_t
                qry_loss_t.backward() # this accumualtes the gradients in a memory efficient way to compute the desired gradients on the meta-loss for each task
                ## append losses
                meta_losses.append(qry_loss_t.detach())
                qry_acc_t = calc_accuracy(mdl=fmodel, X=qry_x_t, Y=qry_y_t)
                meta_accs.append( qry_acc_t )
            meta_loss = meta_loss / nb_tasks
        ## outer update
        #meta_loss.backward()
        if verbose:
            print(f'--> episode = {episode}')
            print(f'meta-loss = {sum(meta_losses)/nb_tasks} \nmeta-accs = {sum(meta_accs)/nb_tasks}')
            print(f'sum(base_model.grad.norm) = {sum([p.grad.norm(2) for p in base_model.parameters()])}')
            print(f'sum(base_model.norm) = {sum([p.norm(2) for p in base_model.parameters()])}')
        assert( base_model.conv1.weight.grad is not None )
        assert( base_model.fc1.weight.grad is not None )
        #outer_opt.step()
        #outer_opt.zero_grad()

def test_maml_directly_good_accumulator_mini_imagenet(verbose=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import OrderedDict

    from pathlib import Path

    from uutils.torch_uu import prepare_data_for_few_shot_learning

    from types import SimpleNamespace

    from meta_learning.meta_learners.maml_meta_learner import MAMLMetaLearner

    import uutils

    uutils

    ## training config
    args = SimpleNamespace()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.mode = 'meta-train'
    args.track_higher_grads = True # if True, during unrolled optimization the graph be retained, and the fast weights will bear grad funcs, so as to permit backpropagation through the optimization process. False during test time for efficiency reasons
    args.copy_initial_weights = False # if False then we train the base models initial weights (i.e. the base model's initialization)
    args.episodes = 5
    args.nb_inner_train_steps = 5
    args.outer_lr = 1e-2
    args.inner_lr = 1e-1 # carefuly if you don't know how to change this one
    # N-way, K-shot, with k_eval points
    args.k_shot, args.k_eval = 5, 15
    args.n_classes = 5 
    C,H,W = [3, 84, 84] # like mini-imagenet
    # loss for tasks
    args.criterion = nn.CrossEntropyLoss() # The input is expected to contain raw, unnormalized scores for each class.
    ## get base model
    kernel_size = 3
    base_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=3,out_channels=7,kernel_size=kernel_size)), # H' = H - k + 1
        ('act', nn.LeakyReLU()),
        ('flatten', Flatten()),
        ('fc1', nn.Linear( (H-kernel_size+1)*(W-kernel_size+1)*7, args.n_classes))
        ]))
    args.bn_momentum = 0.95
    args.bn_eps = 1e-3
    args.grad_clip_mode = 'clip_all_together'
    image_size=args.image_size = H
    #base_model = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum, n_classes=args.n_classes).to(args.device)
    print(f'base_learner = {base_model}')
    ## loading mini-imagenet
    args.n_workers = 4
    args.pin_mem = True
    args.image_size = H
    args.episodes = 5
    args.episodes_val = 4
    args.episodes_test = 3
    args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
    metatrainset_loader, _, _ = prepare_data_for_few_shot_learning(args)
    ## maml-meta-learner
    maml = MAMLMetaLearner(args, base_model)
    ## get outer optimizer (not differentiable nor trainable)
    outer_opt = optim.Adam([{'params': base_model.parameters()}], lr=args.outer_lr)
    for episode, (SQ_x, SQ_y) in enumerate(metatrainset_loader):
        ## get fake support & query data from batch of N tasks
        #spt_x, spt_y, qry_x, qry_y = get_support_query_batch_of_tasks_class_is_task_M_eq_N(args, SQ_x, SQ_y)
        spt_x, spt_y = torch.randn(args.n_classes, args.k_shot, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_shot) for label in range(args.n_classes) ])
        qry_x, qry_y = torch.randn(args.n_classes, args.k_eval, C,H,W), torch.stack([ torch.tensor([label]).repeat(args.k_eval) for label in range(args.n_classes) ])
        ## Acculumate gradients of meta-loss
        losess, meta_accs = maml(spt_x, spt_y, qry_x, qry_y)
        ## outer update
        if verbose:
            print(f'--> episode = {episode}')
            print(f'sum(base_model.grad.norm) = {sum([p.grad.norm(2) for p in base_model.parameters()])}')
            print(f'sum(base_model.norm) = {sum([p.norm(2) for p in base_model.parameters()])}')
        assert(base_model.conv1.weight.grad is not None)
        assert(base_model.fc1.weight.grad is not None)
        outer_opt.step()
        outer_opt.zero_grad()

def import_tests():
    import uutils.emailing
    from uutils.helloworld as helloworld

    out = uutils.helloworld2

    ###

    from meta_learning.base_models.learner_from_opt_as_few_shot_paper import Learner

    learner = Learner()

if __name__ == "__main__":
    #test_maml_directly_good_accumulator_simple()
    #test_maml_meta_learner()
    #test_maml_directly_bad_accumulator()

    test_higher_maml_directly_good_accumulator(verbose=True)
    #test_maml_directly_good_accumulator_mini_imagenet(verbose=True)
    print('Done, all Tests Passed! \a')