from abc import *
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Trainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        pass

    @torch.no_grad()
    def evaluate(self):
        
        accs = list()
        confs = list()
        for test_loader in self.test:
            _acc, _conf = self._evaluate(test_loader)
            accs.append(_acc)
            confs.append(_conf)

        return accs, confs

    def _evaluate(self, loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_conf = 0
        for idx, (data, label) in enumerate(loader):
            
            data = data.to(self.device)
            label = label.to(self.device)

            _pred, feat = self.model(data)
            total_samples += len(data)
            _, pred = torch.max(_pred, 1)
            total_correct += (torch.eq(pred, label)).sum().item()
            conf, _ = torch.max(F.softmax(_pred, 1), 1)
            total_conf += conf.sum().item() 


        return total_correct / total_samples, total_conf/total_samples