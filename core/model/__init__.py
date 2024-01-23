import copy
import torch
from core.model.resnet import SupConResNet as ResNet

def Model_Loader(args):
    
        return ResNet

def Checkpoint_Loader(args, model):
    
    method = args.method 

    if method == "retrain":
        return model

    else:
        checkpoint = torch.load(args.load_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        
        return model


def Unlearn_Checkpoint_Loader(args, model):

    method = args.method

    checkpoint = torch.load(args.load_unlearn_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    return model