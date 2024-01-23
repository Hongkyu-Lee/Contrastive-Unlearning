from torchvision import transforms 

def BasicTransform(normalize, size):
    
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform


def NormalizeTransform(normalize):
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform
