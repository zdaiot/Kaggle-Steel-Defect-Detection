import cv2
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from copy import deepcopy

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout,Rotate, Normalize, Crop, RandomCrop
)

sys.path.append('.')
from utils.visualize import image_with_mask_torch, image_with_mask_numpy
from utils.rle_parse import make_mask


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

        plt.show()


def data_augmentation(original_image, original_mask, crop=False, height=None, width=None):
    """进行样本和掩膜的随机增强
    
    Args:
        original_image: 原始图片
        original_mask: 原始掩膜
    Return:
        image_aug: 增强后的图片
        mask_aug: 增强后的掩膜
    """
    augmentations = Compose([
        HorizontalFlip(p=0.4),
        VerticalFlip(p=0.4),
        ShiftScaleRotate(shift_limit=0.07, rotate_limit=0, p=0.4),
        # 直方图均衡化
        CLAHE(p=0.3),

        # 亮度、对比度
        RandomGamma(gamma_limit=(80, 120), p=0.1),
        RandomBrightnessContrast(p=0.1),
        
        # 模糊
        OneOf([
                MotionBlur(p=0.1),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.3),
        
        OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2)
    ])
    
    if crop:
        # 是否进行随机裁剪
        assert height and width
        crop_aug = RandomCrop(height=height, width=width, always_apply=True)
        crop_sample = crop_aug(image=original_image, mask=original_mask)
        original_image = crop_sample['image']
        original_mask = crop_sample['mask']

    augmented = augmentations(image=original_image, mask=original_mask)
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    return image_aug, mask_aug


if __name__ == "__main__":
    data_folder = "datasets/Steel_data"
    df_path = "datasets/Steel_data/train.csv"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    file_names = df.index.tolist()

    for index in range(len(file_names)):
        image_id, mask = make_mask(index, df)
        image_path = os.path.join(data_folder, 'train_images', image_id)
        image = cv2.imread(image_path)
        image_aug, mask_aug = data_augmentation(image, mask, crop=True, height=256, width=400)
        normalize = Normalize(mean=mean, std=std)
        image = normalize(image=image)['image']
        image_aug = normalize(image=image_aug)['image']

        image_mask = image_with_mask_numpy(deepcopy(image), mask, mean, std)['image']
        image_aug_mask = image_with_mask_numpy(deepcopy(image_aug), mask_aug, mean,std)['image']
        cv2.imshow('image', image_mask)
        cv2.imshow('image_aug', image_aug_mask)
        cv2.waitKey(0)
    pass
