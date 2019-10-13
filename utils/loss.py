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


class SoftBCEDiceLoss(nn.Module):
    """加权BCE+DiceLoss
    """
    def __init__(self, size_average=True, weight=[1.0, 1.0]):
        """
        weight: weight[0]为负类的权重，weight[1]为正类的权重
        """
        super(SoftBCEDiceLoss, self).__init__()
        self.size_average = size_average
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=self.size_average, pos_weight=torch.tensor(weight[0]))
        # self.bce_loss = SoftBceLoss(weight=weight)
        self.softdiceloss = SoftDiceLoss(size_average=self.size_average, weight=weight)
    
    def forward(self, input, target):
        soft_bce_loss = self.bce_loss(input, target)
        soft_dice_loss = self.softdiceloss(input, target)
        loss = 0.85 * soft_bce_loss + 0.15 * soft_dice_loss

        return loss


class MultiClassesSoftBCEDiceLoss(nn.Module):
    def __init__(self, classes_num=4, size_average=True, weight=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], class_weight=[0.3, 0.2, 0.3, 0.2]):
        """

        Args:
            weight: 正负样本权重
            class_weight: 类别间权重
        """
        super(MultiClassesSoftBCEDiceLoss, self).__init__()
        self.classes_num = classes_num
        self.size_average = size_average
        self.class_weight = class_weight
        self.soft_bce_dice_loss = [
            SoftBCEDiceLoss(size_average=self.size_average, weight=weight[0]),
            SoftBCEDiceLoss(size_average=self.size_average, weight=weight[1]),
            SoftBCEDiceLoss(size_average=self.size_average, weight=weight[2]),
            SoftBCEDiceLoss(size_average=self.size_average, weight=weight[3]),
            ]
    
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
            single_class_loss = self.soft_bce_dice_loss[class_index](input_single_class, target_singlt_class)
            loss += self.class_weight[class_index] * single_class_loss

        return loss


class MultiClassBCELoss(nn.Module):
    def __init__(self, class_num=4, class_weight=[0.3, 0.2, 0.3, 0.2]):
        super(MultiClassBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.class_num = class_num
        self.class_weight = class_weight
    
    def forward(self, input, target):
        loss = 0
        for class_index in range(self.class_num):
            input_single_class = input[:, class_index, :, :]
            target_singlt_class = target[:, class_index, :, :]
            single_class_loss = self.bce_loss(input_single_class, target_singlt_class)
            loss += self.class_weight[class_index] * single_class_loss

        return loss


if __name__ == "__main__":
    input = torch.Tensor(4, 4, 256, 1600)
    target = torch.Tensor(4, 4, 256, 1600)
    criterion = MultiClassBCELoss(4)
    loss = criterion(input, target)

