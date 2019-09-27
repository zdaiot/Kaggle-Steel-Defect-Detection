import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import torch
from tqdm import tqdm
import cv2

from create_submission import get_model, post_process
from datasets.steel_dataset import TestDataset, provider


def test_seg_predict(best_threshold, min_size, batch_size, num_workers, mean, std, data_folder, sample_submission_path, model):
    # 加载数据集
    df = pd.read_csv(sample_submission_path)
    dataloader = DataLoader(
        TestDataset(data_folder, df, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # start prediction
    for i, [_, images] in enumerate(tqdm(dataloader)):
        images = images.cuda()
        preds = predict_batch(model, images, best_threshold, min_size)
        pred_show(images, preds, mean, std)


def train_seg_predict(best_threshold, min_size, mean, std, dataloader, model):
    """对训练集和验证集的样本的预测结果进行可视化

    :param best_threshold: 阈值
    :param min_size: 单个掩膜所包含的最小像素个数
    :param mean: 均值
    :param std: 方差
    :param dataloader: 数据加载器
    :param model: 模型
    :return: 无
    """
    for i, [images, targets] in enumerate(tqdm(dataloader)):
        images = images.cuda()
        preds = predict_batch(model, images, best_threshold, min_size)
        pred_show(images, preds, mean, std, targets, True)


def predict_batch(model, images, best_threshold, min_size):
    """对一个batch的样本进行预测

    :param model: 模型
    :param images: 一个batch的样本
    :param best_threshold: 阈值
    :param min_size: 掩膜块最少包含的像素个数
    :return:
    """
    batch_preds = torch.sigmoid(model(images))
    batch_preds = batch_preds.detach().cpu().numpy()    
    for batch_index, preds in enumerate(batch_preds):
        for cls_index, pred in enumerate(preds):
            pred, num = post_process(pred, best_threshold, min_size)
            preds[cls_index] = pred
        batch_preds[batch_index] = preds

    return batch_preds


def pred_show(images, preds, mean, std, targets=None, flag=False):
    """可视化预测结果，与真实类别进行对比

    :param images: 样本
    :param preds: 预测结果
    :param mean: 均值
    :param std: 方差
    :param targets: 真实标定
    :param flag: 是否显示真实标定
    :return: 无
    """
    class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (139, 0, 139)]
    batch_size = images.size(0)
    for index in range(batch_size):
        image = images[index]
        for i in range(3):
            image[i] = image[i] * std[i]
            image[i] = image[i] + mean[i]
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        pred = preds[index]
        for i in range(preds.shape[0]):
            pred_0 = pred[i] * class_color[i][0]
            pred_1 = pred[i] * class_color[i][1]
            pred_2 = pred[i] * class_color[i][2]
            mask = np.stack([pred_0, pred_1, pred_2], axis=2)
            image_pred = image + mask
        cv2.imshow('predict', image_pred)

        if flag:
            target = targets[index]
            for i in range(target.size(0)):
                target_0 = target[i] * class_color[i][0]
                target_1 = target[i] * class_color[i][1]
                target_2 = target[i] * class_color[i][2]
                mask = torch.stack([target_0, target_1, target_2], dim=0)
                image_target = image + mask.permute(1, 2, 0).cpu().numpy()
            cv2.imshow('target', image_target)
        cv2.waitKey(0)


if __name__ == "__main__":
    sample_submission_path = 'datasets/Steel_data/sample_submission.csv'
    test_data_folder = 'datasets/Steel_data/test_images'
    ckpt_path = 'checkpoints/unet_resnet34/unet_resnet34_fold2_best.pth'

    # 设置超参数
    model_name = 'unet_resnet34'
    # initialize test dataloader
    best_threshold = 0.5
    num_workers = 2
    batch_size = 4
    min_size = 3500
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    print('best_threshold', best_threshold)

    model = get_model(model_name, ckpt_path)
    # test_seg_predict(best_threshold, min_size, batch_size, num_workers, mean, std, test_data_folder, sample_submission_path, model)

    data_folder = "datasets/Steel_data"
    df_path = "datasets/Steel_data/train.csv"
    dataloader = provider(data_folder, df_path, mean, std, batch_size, num_workers, 5)
    for fold_index, [train_dataloader, val_dataloader] in enumerate(dataloader):
        train_seg_predict(best_threshold, min_size, mean, std, val_dataloader, model)


