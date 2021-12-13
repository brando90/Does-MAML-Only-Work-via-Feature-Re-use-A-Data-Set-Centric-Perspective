"""
Union of data sets for SL training.
"""
from typing import Union

import torchvision
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path

import torch

from task2vec import Task2Vec
from models import get_model
import datasets
import task_similarity

def get_datasets(root: Union[str, Path], dataset_names: list[str]) -> list[torchvision.datasets]:
    import datasets
    root: Path = Path(root).expanduser() if isinstance(root, str) else root.expanduser()
    data_sets: list[torchvision.datasets] = [datasets.__dict__[name](root=root)[0] for name in dataset_names]
    return data_sets

class UnionDatasets(Dataset):
    """
    todo:
        - bisect into the right data set
        - make sure we are using the right split
    """

    def __init__(self, root: Union[str, Path], dataset_names: list[str], split: str):
        root: Path = Path(root).expanduser() if isinstance(root, str) else root.expanduser()
        # - set fields
        self.root: Path = root
        self.dataset_names: list[str] = dataset_names
        self.split
        # - get data sets
        self.data_sets: list[torchvision.datasets] = get_datasets(dataset_names, root)

    def __len__(self):
        total_numer_of_data_examples: int = sum([len(dataset) for dataset in self.data_sets])
        return total_numer_of_data_examples

    def __getitem__(self, idx: int):
        pass

# - tests

def go_through_hdml1_test():
    # - get data set list
    # dataset_names = ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    # dataset_names = ('mnist',)
    dataset_names = ('stl10', 'letters', 'kmnist')
    root: Path = Path('~/data').expanduser()
    print(f'{root=}')
    dataset_list: list[torchvision.datasets] = [datasets.__dict__[name](root=root)[0] for name in dataset_names]
    print(f'{dataset_list=}')
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(f'{device=}')

    # - get union data loader
    union_datasets: UnionDatasets = UnionDatasets(root, dataset_names)

    # - go through the union data loader


if __name__ == '__main__':
    go_through_hdml1_test()
    print('Done!\n\a')
