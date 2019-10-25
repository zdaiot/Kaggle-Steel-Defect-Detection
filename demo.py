import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import torch
from tqdm import tqdm
import cv2
import os
from config import get_seg_config
from classify_segment import Classify_Segment_Folds_Split, Classify_Segment_Fold
from datasets.steel_dataset import TestDataset, provider


def demo(classify_splits, seg_splits, mean, std, show_truemask_flag, dataloader, model_path, auto_flag, tta_flag, average_strategy):
    '''

    :param classify_splits: 分类模型的折数，类型为字典
    :param seg_splits: 分割模型的折数，类型为字典
    :param mean: 均值
    :param std: 方差
    :param dataloader: 数据加载器
    :param show_truemask_flag: 是否显示真实标定
    :param model_path: 当前模型权重存放的目录
    :param tta_flag: 是否使用tta
    :param average_strategy: 是否使用平均策略
    :return: None
    '''
    if len(classify_splits) == 1 and len(seg_splits) == 1:
        model = Classify_Segment_Fold(classify_splits, seg_splits, model_path, tta_flag=tta_flag, kaggle=0).classify_segment
    else:
        model = Classify_Segment_Folds_Split(classify_splits, seg_splits, model_path, tta_flag=tta_flag, kaggle=0).classify_segment_folds

    # start prediction
    if show_truemask_flag:
        for samples in tqdm(dataloader):
            if len(samples) == 0:
                continue
            images, masks = samples[0], samples[1]
            if len(classify_splits) == 1 and len(seg_splits) == 1:
                results = model(images).detach().cpu().numpy()
            else:
                results = model(images, average_strategy=average_strategy).detach().cpu().numpy()
            pred_show(images, results, mean, std, targets=masks, flag=show_truemask_flag, auto_flag=auto_flag)
    else:
        for fnames, samples in tqdm(dataloader):
            if len(samples) == 0:
                continue
            images, masks = samples[0], samples[1]
            if len(classify_splits) == 1 and len(seg_splits) == 1:
                results = model(images).detach().cpu().numpy()
            else:
                results = model(images, average_strategy=average_strategy).detach().cpu().numpy()
            pred_show(images, results, mean, std, targets=None, flag=show_truemask_flag, auto_flag=auto_flag)


def pred_show(images, preds, mean, std, targets=None, flag=False, auto_flag=False):
    """可视化预测结果，与真实类别进行对比

    :param images: 样本，tensor，[batch_size, 3, h, w]
    :param preds: 预测结果，numpy.array，[batch_size, 4, h, w]
    :param mean: 均值
    :param std: 方差
    :param targets: 真实标定，tensor，[batch_size, 4, h, w]
    :param flag: 是否显示真实标定
    :param auto_flag: 是否使用自动显示
    :return: 无
    """
    class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (139, 0, 139)]
    batch_size = images.size(0)
    for index in range(batch_size):
        # 将图片转换为RGB
        image = images[index]
        for i in range(3):
            image[i] = image[i] * std[i]
            image[i] = image[i] + mean[i]
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        # 叠加预测的掩膜
        pred = preds[index]
        mask = np.zeros([pred.shape[1], pred.shape[2], 3])
        for i in range(pred.shape[0]):
            pred_0 = pred[i] * class_color[i][0]
            pred_1 = pred[i] * class_color[i][1]
            pred_2 = pred[i] * class_color[i][2]
            mask += np.stack([pred_0, pred_1, pred_2], axis=2)
        image_pred = image + mask
        cv2.imshow('predict', image_pred)
        # 叠加真实掩膜
        if flag:
            target = targets[index]
            mask = torch.zeros(3, target.size(1), target.size(2))
            for i in range(target.size(0)):
                target_0 = target[i] * class_color[i][0]
                target_1 = target[i] * class_color[i][1]
                target_2 = target[i] * class_color[i][2]
                mask += torch.stack([target_0, target_1, target_2], dim=0)
            image_target = image + mask.permute(1, 2, 0).cpu().numpy()
            cv2.imshow('target', image_target)
        if auto_flag:
            cv2.waitKey(240)
        else:
            cv2.waitKey(0)


if __name__ == "__main__":
    config = get_seg_config()
    config.batch_size = 1
    # 设置超参数
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    classify_splits = {'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1} # 'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1
    segment_splits = {'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1} # 'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1
    # 在哪一折的验证集上进行验证
    fold = 1
    # 是否使用自动显示
    auto_flag = False
    # 是否显示真实的mask
    show_truemask_flag = True
    tta_flag = True
    average_strategy = False

    # 测试数据集的dataloader
    sample_submission_path = 'datasets/Steel_data/sample_submission.csv'
    test_data_folder = 'datasets/Steel_data/test_images'
    df = pd.read_csv(sample_submission_path)
    test_loader = DataLoader(
        TestDataset(test_data_folder, df, mean, std),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 加载验证集的dataloader
    dataloaders = provider(config.dataset_root, os.path.join(config.dataset_root, 'train.csv'), mean, std, config.batch_size, config.num_workers, config.n_splits)
    valid_loader = dataloaders[fold][1]

    if show_truemask_flag:
        dataloader = valid_loader
    else:
        dataloader = test_loader
    demo(classify_splits, segment_splits, mean, std, show_truemask_flag, dataloader, \
         model_path='./checkpoints/', auto_flag=auto_flag, tta_flag=tta_flag, average_strategy=average_strategy)


