import torch
import torch.nn as nn
from anatome import SimilarityHook

from collections import OrderedDict

#
Din, Dout = 1, 1
mdl1 = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU()),
    ('fc2_l2', nn.Linear(Din, Dout)),
]))
mdl2 = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU()),
    ('fc2_l2', nn.Linear(Din, Dout)),
]))

print(f'is cuda available: {torch.cuda.is_available()}')

with torch.no_grad():
    mu = torch.zeros(Din)
    # std =  1.25e-2
    std = 10
    noise = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
    # mdl2.fc1_l1.weight.fill_(50.0)
    # mdl2.fc1_l1.bias.fill_(50.0)
    mdl2.fc1_l1.weight += noise
    mdl2.fc1_l1.bias += noise

if torch.cuda.is_available():
    mdl1 = mdl1.cuda()
    mdl2 = mdl2.cuda()

hook1 = SimilarityHook(mdl1, "fc1_l1")
hook2 = SimilarityHook(mdl2, "fc1_l1")
mdl1.eval()
mdl2.eval()

# params for doing "good" CCA
iters = 10
num_samples_per_task = 500
size = 8
# start CCA comparision
lb, ub = -1, 1

for _ in range(iters):
    x = torch.torch.distributions.Uniform(low=-1, high=1).sample((num_samples_per_task, 1))
    if torch.cuda.is_available():
        x = x.cuda()
    y1 = mdl1(x)
    y2 = mdl2(x)
    print(f'y1 - y2 = {(y1-y2).norm(2)}')
print('about to do cca')
dist = hook1.distance(hook2, size=size)
print('cca done')
print(f'cca dist = {dist}')
print('--> Done!\a')
