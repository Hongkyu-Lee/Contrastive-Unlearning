import torch
import wandb
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from core.trainer.base import Trainer
from core.data import Data_Loader


# Normal training


class NormalTrainer(Trainer):

    def __init__(self, model:Module, trainloader:DataLoader, testloader:DataLoader, args):
        self.model = model
        self.tr = trainloader
        self.test = testloader
        self.epoch_max = args.epochs+1
        self.epoch_current = 1
        self.optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.w_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = args.device

        self.model.to(self.device)   
        self.scheduler = CosineAnnealingLR(self.optim, T_max = self.epoch_max)

    def fit(self, e):
        
        self.model.train()
        avg_loss = 0.
        total_sample = 0
        total_correct = 0

        for idx, (data, label) in enumerate(self.tr):
            
            self.optim.zero_grad()
            data = data.to(self.device)
            label = label.to(self.device)
            pred, feat = self.model(data)
            loss = self.criterion(pred, label)
            loss.backward()
            self.optim.step()

            avg_loss += loss.item()

            _, _pred = torch.max(pred, 1)
            total_sample += label.size(0)
            total_correct += (torch.eq(_pred, label)).sum().item()

        self.epoch_current+=1
        self.scheduler.step()

        return avg_loss / len(self.tr), total_correct / total_sample

    @torch.no_grad()
    def evaluate(self):
        accs, confs = super().evaluate()
        return accs[0], confs[0]