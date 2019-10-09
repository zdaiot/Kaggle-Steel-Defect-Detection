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


# reference https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429#latest-588288
class SoftDiceLoss(nn.Module):
    """二分类加权dice损失
    """
    def __init__(self, size_average=True, weight=[0.2, 0.8]):
        """
        weight: 各类别权重
        """
        super(SoftDiceLoss, self).__init__()
        self.size_average = size_average
        self.weight = torch.FloatTensor(weight)
    
    def forward(self, logit_pixel, truth_pixel):
        batch_size = len(logit_pixel)
        logit = logit_pixel.view(batch_size, -1)
        truth = truth_pixel.view(batch_size, -1)
        assert(logit.shape == truth.shape)

        loss = self.soft_dice_criterion(logit, truth)

        if self.size_average:
            loss = loss.mean()
        return loss

    def soft_dice_criterion(self, logit, truth):
        batch_size = len(logit)
        probability = torch.sigmoid(logit)

        p = probability.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        # 向各样本分配所属类别的权重
        w = truth.detach()
        self.weight = self.weight.type_as(logit)
        w = w * (self.weight[1] - self.weight[0]) + self.weight[0]

        p = w * (p*2 - 1)  #convert to [0,1] --> [-1, 1]
        t = w * (t*2 - 1)

        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - 2 * intersection/union

        loss = dice
        return loss


# reference https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429#latest-588288
class SoftBceLoss(nn.Module):
    """二分类交叉熵加权损失
    """
    def __init__(self, weight=[0.25, 0.75]):
        super(SoftBceLoss, self).__init__()
        self.weight = weight

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        if self.weight:
            pos = (truth>0.5).float()
            neg = (truth<0.5).float()
            # pos_weight = pos.sum().item() + 1e-12
            # neg_weight = neg.sum().item() + 1e-12
            # loss = (self.weight[0]*pos*loss/pos_weight + self.weight[1]*neg*loss/neg_weight).sum()
            loss = (self.weight[1]*pos*loss + self.weight[0]*neg*loss).mean()
        else:
            loss = loss.mean()
        return loss


class SoftBCEDiceLoss(nn.Module):
    """加权BCE+DiceLoss
    """
    def __init__(self, size_average=True, weight=[0.2, 0.8]):
        """
        weight: weight[0]为负类的权重，weight[1]为正类的权重
        """
        super(SoftBCEDiceLoss, self).__init__()
        self.size_average = size_average
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=self.size_average, pos_weight=torch.tensor(self.weight[1]))
        # self.bce_loss = SoftBceLoss(weight=weight)
        self.softdiceloss = SoftDiceLoss(size_average=self.size_average, weight=weight)
    
    def forward(self, input, target):
        soft_bce_loss = self.bce_loss(input, target)
        soft_dice_loss = self.softdiceloss(input, target)
        loss = 0.7 * soft_bce_loss + 0.3 * soft_dice_loss

        return loss


class MultiClassesSoftBCEDiceLoss(nn.Module):
    def __init__(self, classes_num=4, size_average=True, weight=[0.2, 0.8]):
        super(MultiClassesSoftBCEDiceLoss, self).__init__()
        self.classes_num = classes_num
        self.size_average = size_average
        self.weight = weight
        self.soft_bce_dice_loss = SoftBCEDiceLoss(size_average=self.size_average, weight=self.weight)
    
    def forward(self, input, target):
        """
        Args:
            input: tensor, [batch_size, classes_num, height, width]
            target: tensor, [batch_size, classes_num, height, width]
        """
        loss = 0
        for class_index in range(self.classes_num):
            input_single_class = input[:, class_index, :, :]
            target_singlt_class = target[:, class_index, :, :]
            single_class_loss = self.soft_bce_dice_loss(input_single_class, target_singlt_class)
            loss += single_class_loss
        
        loss /= self.classes_num

        return loss


if __name__ == "__main__":
    input = torch.Tensor(4, 4, 256, 1600)
    target = torch.Tensor(4, 4, 256, 1600)
    criterion = MultiClassesSoftBCEDiceLoss(4, True)
    loss = criterion(input, target)

