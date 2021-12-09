#!/home/miranda9/miniconda/envs/automl-meta-learning/bin/python3.7
#PBS -V
#PBS -M brando.science@gmail.com
#PBS -m abe
#PBS -l nodes=1:ppn=4:gpus=1,walltime=96:00:00


import torch

import torchvision.transforms as transforms

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from tqdm import tqdm

from pathlib import Path

meta_split = 'train'
data_path = Path('~/data/').expanduser()
dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
print(f'len normal = {len(dataloader)}')

num_batches = 10
with tqdm(dataloader, total=num_batches) as pbar:
    for batch_idx, batch in enumerate(pbar):
        train_inputs, train_targets = batch["train"]
        print(train_inputs.size())
        # print(batch_idx)
        if batch_idx >= num_batches:
            break

print('success\a')