import torch

def Sampler_Loader(args, datasets, is_train=False):
    samplers = list()
    _len = len(datasets)
    
    if args.method == "contrastive" and is_train:
        samplers.append(None)
        samplers.append(
            torch.utils.data.RandomSampler(datasets[1], replacement=True))
        for _ in range(_len-2):
            samplers.append(None)
        return samplers
    
    else:
        return [None for _ in range(_len)]