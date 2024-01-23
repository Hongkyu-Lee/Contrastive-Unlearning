import copy
import torch
import wandb
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn import functional as F
from core.trainer.base import Trainer
# from core.data import unlearn_batch


class UnlearnTrainer(Trainer):
    
    def __init__(self, model:Module, trainloader:list, testloaders:list, args):
        self.model = model
        self.device=args.device
        self.retain = trainloader[1]
        self.unlearn = trainloader[0]
        self.orig_model = copy.deepcopy(model).to(self.device)
        self.optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        self.batch_size = args.batch_size
        self.model.to(self.device)
        self.temperature = args.temp
        self.base_temperature =args.temp
        self.num_retain_sample = args.retain_sampling_freq
        self.CT_ratio = args.CT_ratio
        self.CE_ratio = args.CE_ratio
        self.unlearn_type = args.unlearn_type

        self.test = testloaders
        self.terminated = False

        if args.loss_ver == 1:
            self.Loss = self.Loss_v1
        elif args.loss_ver == 2:
            self.Loss = self.Loss_v2
        else:
            raise ValueError(f"Loss select: {args.loss_ver}")

        self.loss_ver = args.loss_ver

    def fit(self, epoch):

        self.model.train()

        for u_idx, (_u_img, _u_label) in enumerate(self.unlearn):
            
            retain_iter = iter(self.retain)

            avg_mse = 0.0
            
            for i in range(self.num_retain_sample):
                
                self.optim.zero_grad()

                rt_img, rt_label = next(retain_iter)
        
                u_img, u_label = _u_img.clone().detach(), _u_label.clone().detach()
                _batch_size = u_img.shape[0]

                u_img = u_img.to(self.device)
                rt_img = rt_img[:_batch_size].to(self.device)
                u_label = u_label.to(self.device)
                rt_label = rt_label[:_batch_size].to(self.device)

                imgs = torch.cat([u_img, rt_img], dim=0)       
                pred, feat = self.model(imgs)
                
                u_pred, rt_pred = torch.split(pred, [_batch_size, _batch_size], dim=0)
                u_feat, rt_feat = torch.split(feat, [_batch_size, _batch_size], dim=0)

                loss = self.Loss(u_pred, rt_pred, u_feat, rt_feat, u_label, rt_label)

                loss.backward()
                self.optim.step()

                avg_mse += F.mse_loss(u_feat, rt_feat).item()

        unlearn_acc, unlearn_conf = self._evaluate(self.unlearn)

        out_dict = {
            "unlearn_acc": unlearn_acc,
        }

        return out_dict
            
    def evaluate(self):
        return super().evaluate()

    def Loss_v1(self, u_pred, rt_pred, u_feat, rt_feat, u_label, rt_label):

        """
        Loss types

        Type 1: CT only
            Anchor :  unlearning images
            positive: retain images with same labels
            Negative: retain images with different labels

        """

        n_u = u_label.size(0)
        n_rt = rt_label.size(0)

        u_label = u_label.contiguous().view(-1, 1)
        rt_label = rt_label.contiguous().view(-1, 1)

        mask = torch.eq(u_label, rt_label.T)
        neg_mask = (~mask).clone().float()
        neg_count = torch.sum(mask).item()

        pos_mask = (mask).clone().float()
        neg_add_mask = (~(pos_mask.sum(1).bool())).int().contiguous().view(-1, 1)
        
        orig_logits = torch.matmul(u_feat, rt_feat.T)

        logits = torch.div(orig_logits, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()

        exp_logits = torch.exp(logits) + 1e-20

        neg_logits = logits * neg_mask
        pos_logits = (exp_logits * pos_mask).sum(1, keepdim=True)
        if self.unlearn_type == "single_class":
            pos_logits += neg_add_mask
            

        log_prob = neg_logits - torch.log(pos_logits)
        mean_log_prob = log_prob.sum(1) / neg_mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        loss = loss.mean()

        return loss

    def Loss_v2(self, u_pred, rt_pred, u_feat, rt_feat, u_label, rt_label):
        """

        Type 2: CT + CE
        CE on retain samples only
    
        """

        CT_loss = self.Loss_v1(u_pred, rt_pred, u_feat,
                               rt_feat, u_label, rt_label)
        
        CE_loss = F.cross_entropy(rt_pred, rt_label)
        
        loss = CT_loss * self.CT_ratio + CE_loss * self.CE_ratio

        return loss
