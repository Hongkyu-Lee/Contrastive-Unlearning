import torch

def select_optim(name:str, model:torch.nn.Modules):
    if name == "adam":
        return torch.optim.Adam(model.parameters())
    if name == "sgd":
        return torch.optim.SGD(model.parameters())