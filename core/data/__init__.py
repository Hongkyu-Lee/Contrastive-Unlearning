import torch
from core.data.datasets import DATASETS, CustomImageDataset
from core.data.transform import BasicTransform
from core.data.transform import NormalizeTransform
from core.data.loader import UNLEARN_DATA
from core.data.sampler import Sampler_Loader

def Data_Loader(args):

    # Provide dataloaders for train, test and unlean set

    train_data, test_data = DATASETS[args.dataset](args)
    train_data, test_data = UNLEARN_DATA[args.unlearn_type](args, train_data, test_data)
    train_samplers = Sampler_Loader(args, train_data)
    test_samplers = Sampler_Loader(args, test_data)

    trainloaders = list()
    testloaders = list()

    for data, sampler in zip(train_data, train_samplers):
        if type(data) == type(dict()):
            _loader = dict()
            for k, v in data.items():
                _loader[k] = torch.utils.data.DataLoader(v, batch_size=args.batch_size,
                                    shuffle=None, num_workers=args.num_workers,
                                    pin_memory=True, sampler=sampler)
            trainloaders.append(
                _loader
            )
        else:

            trainloaders.append(
                torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                            shuffle=None, num_workers=args.num_workers,
                                            pin_memory=True, sampler=sampler
            ))

    for data, sampler in zip(test_data, test_samplers):
        if type(data) == type(dict()):
            _loader = dict()
            for k, v in data.items():
                _loader[k] = torch.utils.data.DataLoader(v, batch_size=args.batch_size,
                                    shuffle=None, num_workers=args.num_workers,
                                    pin_memory=True, sampler=sampler)
            testloaders.append(
                _loader
            )
        else:

            testloaders.append(
                torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                            shuffle=None, num_workers=args.num_workers,
                                            pin_memory=True, sampler=sampler
            ))    

    return trainloaders, testloaders
