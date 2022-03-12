import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd

from ..builder import LOSSES
from .utils import weight_reduce_loss

@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=2.0,
                 alpha=0.25,):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eflv2
        self.gamma = gamma
        self.alpha = alpha
        self.category_counts = pd.read_csv('category_counts.csv')
        self.total_count = sum(self.category_counts.counts)
        self.s = torch.Tensor(self.total_count/self.category_counts.counts)

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction='mean',
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pt = (1 - cls_score) * target + cls_score * (1 - target)
        
        map_val = 1 - self.pos_neg.detach()
        
        # Compute score factor
        
        dy_gamma = self.gamma + self.s * map_val
        
        # focusing factor
        
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)
        
        # weighting factor
        
        wf = ff / self.gamma
        
        # Alpha
        
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # ce_loss        
        
        ce_loss = -torch.log(pt)
        
        cls_loss = ce_loss * torch.pow((1 - pt), ff.detach()) * wf.detach() * alpha_t
                
        cls_loss = weight_reduce_loss(cls_loss, weight, reduction, avg_factor)
    
        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes)
            self._neg_grad = cls_score.new_zeros(self.num_classes)
            neg_w = cls_score.new_ones((self.n_i, self.n_c))
            pos_w = cls_score.new_ones((self.n_i, self.n_c))
        else:
            # the negative weight for objectiveness is always 1
            neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w