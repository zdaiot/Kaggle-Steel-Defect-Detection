import torch
import numpy as np


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(logit, truth, threshold=0.5):
    batch_size, num_class, H, W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size, num_class, -1)
        truth = truth.view(batch_size, num_class, -1)

        probability = torch.sigmoid(logit)
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives
        # 各个类别预测正确的正样本、负样本数目
        tp = tp.sum(dim=[0, 2])
        tn = tn.sum(dim=[0, 2])
        num_pos = t.sum(dim=[0, 2])
        num_neg = batch_size * H * W - num_pos
        # 预测正确的正样本和负样本的数目
        tp = tp.data.cpu().numpy().sum()
        tn = tn.data.cpu().numpy().sum()
        # 正样本、负样本的数目
        num_pos = num_pos.data.cpu().numpy().sum()
        num_neg = num_neg.data.cpu().numpy().sum()

        # tp = np.nan_to_num(tp / (num_pos + 1e-12), 0)
        # tn = np.nan_to_num(tn / (num_neg + 1e-12), 0)

        # tp = list(tp)
        # num_pos = list(num_pos)

    return tn, tp, num_neg, num_pos


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self):
        self.base_threshold = 0.5
        self.true_negative = []
        self.true_poisitive = []
        self.number_negative = []
        self.number_positive = []

    def update(self, targets, outputs):
        tn, tp, num_neg, num_pos = metric(outputs, targets, self.base_threshold)
        self.true_negative.append(tn)
        self.true_poisitive.append(tp)
        self.number_negative.append(num_neg)
        self.number_positive.append(num_pos)

    def get_metrics(self):
        # 预测正确的样本的数目
        tn = np.sum(self.true_negative)
        tp = np.sum(self.true_poisitive)
        # 负样本和正样本各自的数目
        num_neg = np.sum(self.number_negative)
        num_pos = np.sum(self.number_positive)
        # 正负样本各自的准确率和总的准确率
        neg_accuracy = tn / (num_neg + 1e-12)
        pos_accuracy = tp / (num_pos + 1e-12)
        accuracy = (tn + tp) / (num_neg + num_pos)

        return neg_accuracy, pos_accuracy, accuracy