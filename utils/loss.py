import torch
from torch import nn
import torch.nn.functional as F


class ClassifyLoss(nn.Module):
    def __init__(self, weight=None):
        """
        Args:
            weight: 正负样本的权重
        """
        super(ClassifyLoss, self).__init__()
        self.weight = weight

    def forward(self, logit, truth):
        batch_size, num_class, H, W = logit.shape
        logit = logit.view(batch_size, num_class)
        truth = truth.view(batch_size, num_class)
        assert(logit.shape == truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        if self.weight is None:
            loss = loss.mean()

        else:
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()
            pos_sum = pos.sum().item() + 1e-12
            neg_sum = neg.sum().item() + 1e-12
            loss = (self.weight[1] * pos * loss / pos_sum + self.weight[0] * neg * loss / neg_sum).sum()
            # raise NotImplementedError

        return loss
