# coding: utf-8

import os
import cv2
import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset, sampler
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
from utils.rle_parse import mask2rle, make_mask
warnings.filterwarnings("ignore")


# Dataset
class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images",  image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1) # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def provider(
    data_folder,
    df_path,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
    n_splits=0,
):
    """返回数据加载器

    Args:
        data_folder: 数据集根目录
        df_path: csv文件路径
        mean: 数据集各通道均值
        std: 数据集各通道标准差
        batch_size
        num_workers
        n_split: 交叉验证折数，为1时不使用交叉验证
    
    Return:
        dataloadrs: list，该list中的每一个元素为list，元素list中保存训练集和验证集
    """
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    
    # 将数据集划分为n_split份
    train_dfs = list()
    val_dfs = list()
    if n_splits != 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)
        # 按照样本包含的损伤的类别的数目进行分层
        for train_df_index, val_df_index in skf.split(df, df['defects']):
            train_dfs.append(df.loc[df.index[train_df_index]])
            val_dfs.append(df.loc[df.index[val_df_index]])
    else:        
        df_temp = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
        train_dfs.append(df_temp[0])
        val_dfs.append(df_temp[1])
    # 生成dataloader
    dataloaders = list()
    for df_index, (train_df, val_df) in enumerate(zip(train_dfs, val_dfs)):
        train_dataset = SteelDataset(train_df, data_folder, mean, std, 'train')
        val_dataset = SteelDataset(val_df, data_folder, mean, std, 'val')
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=True
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=True
        )
        dataloaders.append([train_dataloader, val_dataloader])

    return dataloaders


if __name__ == "__main__":
    data_folder = "/home/apple/program/MXQ/Competition/Kaggle/Steal-Defect/Kaggle-Steel-Defect-Detection/datasets/Steel_data"
    df_path = "/home/apple/program/MXQ/Competition/Kaggle/Steal-Defect/Kaggle-Steel-Defect-Detection/datasets/Steel_data/train.csv"
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    batch_size = 8
    num_workers = 4
    n_splits = 1
    dataloader = provider(data_folder, df_path, mean, std, batch_size, num_workers, n_splits)
    for fold_index, [train_dataloader, val_dataloader] in enumerate(dataloader):
        train_bar = tqdm(train_dataloader)
        class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 139]]
        for images, targets in train_bar:
            image = images[0]
            for i in range(3):
                image[i] = image[i] * std[i]
                image[i] = image[i] + mean[i]

            target = targets[0]
            for i in range(target.size(0)):
                target_0 = target[i] * class_color[i][0]
                target_1 = target[i] * class_color[i][1]
                target_2 = target[i] * class_color[i][2]
                mask = torch.stack([target_0, target_1, target_2], dim=0)
                image += mask
            image = image.permute(1, 2, 0).numpy()
            cv2.imshow('win', image)
            cv2.waitKey(0)

            pass
