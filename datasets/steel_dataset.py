# coding: utf-8

import os
import cv2
import warnings
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision import transforms
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
import sys

sys.path.append('.')
from utils.data_augmentation import data_augmentation
from utils.rle_parse import mask2rle, make_mask
from utils.visualize import image_with_mask_torch
import pickle
warnings.filterwarnings("ignore")


# Dataset Segmentation
class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        super(SteelDataset, self).__init__()
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images",  image_id)
        img = cv2.imread(image_path)
        img, mask = self.transforms(self.phase, img, mask, self.mean, self.std)
        mask = mask.permute(2, 0, 1)
        return img, mask

    def __len__(self):
        return len(self.fnames)


# Dataset Classification
class SteelClassDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        super(SteelClassDataset, self).__init__()
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images",  image_id)
        img = cv2.imread(image_path)
        img, mask = self.transforms(self.phase, img, mask, self.mean, self.std)
        mask = mask.permute(2, 0, 1) # 4x256x1600
        mask = mask.view(mask.size(0), -1)
        mask = torch.sum(mask, dim=1)
        mask = mask > 0

        return img, mask.float()

    def __len__(self):
        return len(self.fnames)


class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def augmentation(image, mask):
    """进行数据增强
    Args:
        image: 原始图像
        mask: 原始掩膜
    Return:
        image_aug: 增强后的图像，Image图像
        mask: 增强后的掩膜，Image图像
    """
    image_aug, mask_aug = data_augmentation(image, mask)
    image_aug = Image.fromarray(image_aug)

    return image_aug, mask_aug


def get_transforms(phase, image, mask, mean, std):

    if phase == 'train':
        image, mask = augmentation(image, mask)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    transform_compose = transforms.Compose([to_tensor, normalize])
    image = transform_compose(image)
    mask = torch.from_numpy(mask)

    return image, mask


def provider(
    data_folder,
    df_path,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
    n_splits=0,
):
    """返回数据加载器，用于分割模型

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


def classify_provider(
    data_folder,
    df_path,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
    n_splits=0,
):
    """返回数据加载器，用于分类模型

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
        train_dataset = SteelClassDataset(train_df, data_folder, mean, std, 'train')
        val_dataset = SteelClassDataset(val_df, data_folder, mean, std, 'val')
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
    data_folder = "Steel_data"
    df_path = "Steel_data/train.csv"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 8
    num_workers = 4
    n_splits = 1
    # 测试分割数据集
    dataloader = provider(data_folder, df_path, mean, std, batch_size, num_workers, n_splits)
    for fold_index, [train_dataloader, val_dataloader] in enumerate(dataloader):
        train_bar = tqdm(train_dataloader)
        class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 139]]
        for images, targets in train_bar:
            image = images[0]
            target = targets[0]
            image = image_with_mask_torch(image, target, mean, std)['image']
            cv2.imshow('win', image)
            cv2.waitKey(0)
    class_dataloader = classify_provider(data_folder, df_path, mean, std, batch_size, num_workers, n_splits)
    # 测试分类数据集
    for fold_index, [train_dataloader, val_dataloader] in enumerate(class_dataloader):
        train_bar = tqdm(train_dataloader)
        class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (139, 0, 139)]
        for images, targets in train_bar:
            image = images[0]
            for i in range(3):
                image[i] = image[i] * std[i]
                image[i] = image[i] + mean[i]  
            image = image.permute(1, 2, 0).numpy()            
            target = targets[0]
            position_x = 10
            for i in range(target.size(0)):
                color = class_color[i]
                position_x += 50
                position = (position_x, 50)
                if target[i] != 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    image = cv2.putText(image, str(i), position, font, 1.2, color, 2)   
            cv2.imshow('win', image)
            cv2.waitKey(60)

    pass
