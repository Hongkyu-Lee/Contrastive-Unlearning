import torch
from torch.nn import Module as Module
from core.trainer.base import Trainer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

class Retrain(Trainer):
    def __init__(self, model:Module, trainloaders:list, testloaders:list, args) -> None:
        super().__init__()

        self.unlearn = trainloaders[0]
        self.retain = trainloaders[1]
        self.test = testloaders

        self.model = model
        self.device =  args.device

        self.optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.w_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = CosineAnnealingLR(self.optim, T_max = args.unlearn_epoch)

        self.model.to(self.device)

    def fit(self, e):

        self.model.train()
        retain_avg_loss = 0.
        total_sample = 0
        total_correct = 0

        for idx, (data, label) in enumerate(self.retain):
            
            self.optim.zero_grad()
            data = data.to(self.device)
            label = label.to(self.device)
            pred, feat = self.model(data)
            loss = self.criterion(pred, label)
            loss.backward()
            self.optim.step()

            retain_avg_loss += loss.item()

            _, _pred = torch.max(pred, 1)
            total_sample += label.size(0)
            total_correct += (torch.eq(_pred, label)).sum().item()

        self.scheduler.step()

        retain_avg_loss /= len(self.retain)
        retain_acc = total_correct / total_sample


        self.model.eval()
        unlearn_loss = 0.
        total_sample = 0
        total_correct = 0
        
        for idx, (data, label) in enumerate(self.unlearn):

            self.optim.zero_grad()
            data = data.to(self.device)
            label = label.to(self.device)
            pred, feat = self.model(data)
            loss = self.criterion(pred, label)
            unlearn_loss += loss.item()
    
            _, _pred = torch.max(pred, 1)
            total_sample += label.size(0)
            total_correct += (torch.eq(_pred, label)).sum().item()      

        unlearn_loss /= len(self.unlearn)
        unlearn_acc = total_correct / total_sample

        out_dict = {
            "loss": retain_avg_loss,
            "unlearn_acc": unlearn_acc
        }

        return out_dict 
    
    def evaluate(self):
        return super().evaluate()
    