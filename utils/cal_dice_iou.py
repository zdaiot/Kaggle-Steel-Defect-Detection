import torch
import numpy as np


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def compute_dice_class(preds, targs):
    ''' 计算某一类的dice值，注意 preds 要在外部经过sigmoid函数

    Args:
        preds: 维度为[batch_size, 1, height, width]，每一个值表示预测出的属于该类的概率
        targs: 维度为[batch_size, 1, height, width]，每一个值表示真实是否属于该类
    
    Return: 
        计算出的dice值
    '''
    n = preds.shape[0]  # batch size为多少
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    # preds, targs = preds.to(self.device), targs.to(self.device)
    preds, targs = preds.cpu(), targs.cpu()

    # tensor之间按位相成，求两个集合的交(只有1×1等于1)后。按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的交集大小
    intersect = (preds * targs).sum(-1).float()
    # tensor之间按位相加，求两个集合的并。然后按照第二个维度求和，得到[batch size]大小的tensor，每一个值代表该输入图片真实类标与预测类标的并集大小
    union = (preds + targs).sum(-1).float()
    '''
    输入图片真实类标与预测类标无并集有两种情况：第一种为预测与真实均没有类标，此时并集之和为0；第二种为真实有类标，但是预测完全错误，此时并集之和不为0;

    寻找输入图片真实类标与预测类标并集之和为0的情况，将其交集置为1，并集置为2，最后还有一个2*交集/并集，值为1；
    其余情况，直接按照2*交集/并集计算，因为上面的并集并没有减去交集，所以需要拿2*交集，其最大值为1
    '''
    u0 = union == 0
    intersect[u0] = 1
    union[u0] = 2
    
    return (2. * intersect / union).mean()