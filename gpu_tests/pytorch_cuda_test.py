import torch
import torch.nn as nn

from collections import OrderedDict

# import torchvision.transforms as transforms
# from torch.models.custom_layers import Flatten

class Flatten(nn.Module):
    def forward(self, input):
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out  # (batch_size, *size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{torch.cuda.is_available()}')
print(f'Device = {device}')
assert(torch.cuda.is_available())

data = torch.randn(2, 3, 84, 84).to(device)

out_features = 64
model = nn.Sequential(OrderedDict([
    ('features', nn.Sequential(OrderedDict([('flatten', Flatten())]))),
    ('cls', torch.nn.Linear(in_features=84 * 84 * 3, out_features=out_features, bias=True))
]))
model = nn.Sequential(OrderedDict([('model', model)])).to(device)

out = model(data)

print(out.sum())
print('Success! Your code works with gpu')





