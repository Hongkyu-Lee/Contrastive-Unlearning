import copy
import torch
import numpy as np
from torchvision import transforms
from core.data.transform import NormalizeTransform
from core.data.datasets import CustomImageDataset

UNLEARN_DATA = {}

def _add(fn):
    UNLEARN_DATA[fn.__name__] = fn
    return fn

@_add
def random_sample(args, train_data, test_data):
    num_unlearn = int(args.num_unlearn)
    unlearn_indices = np.arange(num_unlearn)
    retain_indices = np.arange(num_unlearn, len(train_data))

    unlearn_x = train_data.data[unlearn_indices]
    try:
        unlearn_y = (np.array(train_data.targets)[unlearn_indices])
    except:
        unlearn_y = (np.array(train_data.labels)[unlearn_indices])

    retain_x = train_data.data[retain_indices]
    try:
        retain_y = (np.array(train_data.targets)[retain_indices])
    except:
        retain_y = (np.array(train_data.labels)[retain_indices])

    unlearn_dataset = CustomImageDataset(
        x=unlearn_x, y=unlearn_y, transform=train_data.transform
    )
    retain_dataset = CustomImageDataset(
        x=retain_x, y=retain_y, transform=train_data.transform
    )

    return (unlearn_dataset, retain_dataset),  (test_data,)

@_add
def single_class(args, train_data, test_data):

        
    if args.method == "boundary_shrink" or args.method == "boundary_expand":
        traindata, testdata = bu_unlearn_loader(args, train_data, test_data, -1)

    elif args.method == "scrub":
        traindata, testdata = scrub_unlearn_loader(args, train_data, test_data, -1)

    elif args.method == "unsir":
        traindata, testdata = unsir_unlearn_loader(args, train_data, test_data, -1)
    
    else:
        unlearn_class = args.unlearn_class
        num_unlearn = -1
        traindata, testdata = unlearn_class_loader(train_data, test_data, unlearn_class, num_unlearn)

    return traindata, testdata

def unlearn_class_loader(train_data, test_data, unlearn_class, num_unlearn):

    traindata = list()
    testdata = list()

    try:
        targets = train_data.targets
    except:
        targets = train_data.labels

    unlearn_idx = torch.where(torch.tensor(targets) == unlearn_class)[0]

    if num_unlearn < 0:
        # Return the entire class
        unlearn_idx = unlearn_idx[:num_unlearn] 

    unlearn_mask = torch.zeros(len(train_data), dtype=torch.bool)
    unlearn_mask[unlearn_idx] = True
    
    retain_mask = ~(unlearn_mask.clone())

    unlearn_data = train_data.data[unlearn_mask]
    unlearn_label = (np.array(targets))[unlearn_mask]
    retain_data = train_data.data[retain_mask]
    retain_label =  (np.array(targets))[retain_mask]

    _transform = train_data.transform

    unlearn_dataset = CustomImageDataset(unlearn_data, unlearn_label, transform=_transform)
    retain_dataset = CustomImageDataset(retain_data, retain_label, transform=_transform)
    
    traindata.append(unlearn_dataset)
    traindata.append(retain_dataset)

    try:
        targets = test_data.targets
    except:
        targets = test_data.labels

    # Do same for test dataset
    unlearn_idx = torch.where(torch.tensor(targets) == unlearn_class)[0]
    unlearn_mask = torch.zeros(len(test_data), dtype=torch.bool)
    unlearn_mask[unlearn_idx] = True
    
    retain_mask = ~(unlearn_mask.clone())

    test_unlearn_class_data = test_data.data[unlearn_mask]
    test_unlearn_class_label = (np.array(targets))[unlearn_mask]
    test_retain_class_data = test_data.data[retain_mask]
    test_retain_class_label =  (np.array(targets))[retain_mask]

    _transform = test_data.transform

    test_unlearn_dataset = CustomImageDataset(test_unlearn_class_data, test_unlearn_class_label,
                                            transform=_transform)
    test_retain_dataset = CustomImageDataset(test_retain_class_data, test_retain_class_label,
                                            transform=_transform)
    
    testdata.append(test_unlearn_dataset)
    testdata.append(test_retain_dataset)

    
    
    if num_unlearn > 0: # Class subset unlearn
        testdata.append(test_data)
    
    return traindata, testdata

    
def unlearn_batch(data, batch_size):
    
    img, label = data
    imgs = img.repeat(batch_size, 1, 1, 1)
    labels = label.repeat(batch_size)

    return imgs, labels
