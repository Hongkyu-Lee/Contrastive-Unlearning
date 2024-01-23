import torch
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
from core.data.transform import BasicTransform


## Standard datasets

DATASETS = {}

def _add(fn):
    DATASETS[fn.__name__] = fn
    return fn

@_add
def cifar_10(args):
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean, std)
    train_transform = BasicTransform(normalize, args.cropsize)
    train_data = datasets.CIFAR10(root=args.datapath, train=True,
                                  transform=train_transform,
                                  download=True)
    
    test_data  = datasets.CIFAR10(root=args.datapath, train=False,
                                  transform=train_transform,
                                  download=True)
    
    return train_data, test_data

@_add
def svhn(args):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean, std)
    train_transform = BasicTransform(normalize, args.cropsize)
    train_data = datasets.SVHN(root=args.datapath, split="train",
                                  transform=train_transform,
                                  download=True)
    test_data  = datasets.SVHN(root=args.datapath, split="test",
                                  transform=train_transform,
                                  download=True)
    
    return train_data, test_data
    
    
#### Custom Dataset class

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform, idx=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.idx = idx

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = torch.tensor(self.y[idx])

        if self.transform:
            try:
                x = self.transform(x)
            except:
                x = self.transform(np.transpose(x, (1, 2, 0)))
        return x, y
    