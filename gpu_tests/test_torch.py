#!/home/miranda9/miniconda/envs/automl-meta-learning/bin/python3.7
#PBS -V
#PBS -M brando.science@gmail.com
#PBS -m abe
#PBS -l nodes=1:ppn=4:gpus=1,walltime=96:00:00


import torch
import torchmeta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print(torchmeta)

x = torch.randn(2, 3)
w = torch.randn(3, 5)
y = x@w

print(y.sum())
