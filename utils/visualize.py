# 可视化操作
import torch
import numpy as np


def image_with_mask_torch(image, target, mean=None, std=None, mask_only=False):
    """返回numpy形式的样本和掩膜
    :param image: 样本，tensor
    :param target: 掩膜，tensor
    :param mean: 样本均值
    :param std: 样本标准差
    :param mask_only: bool，是否只返回掩膜
    """
    class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 139]]
    if mean and std:
        for i in range(3):
            image[i] = image[i] * std[i]
            image[i] = image[i] + mean[i]
    mask = torch.zeros(3, target.size(1), target.size(2))
    for i in range(target.size(0)):
        target_0 = target[i] * class_color[i][0]
        target_1 = target[i] * class_color[i][1]
        target_2 = target[i] * class_color[i][2]
        mask += torch.stack([target_0, target_1, target_2], dim=0)
    image += mask

    pair = {'mask': mask.permute(1, 2, 0).numpy()}
    if not mask_only:
        pair['image'] = image.permute(1, 2, 0).numpy()

    return pair


def image_with_mask_numpy(image, target, mean=None, std=None, mask_only=False):
    """返回numpy形式的样本和掩膜
    :param image: 样本，numpy
    :param target: 掩膜，numpy
    :param mean: 样本均值
    :param std: 样本标准差
    :param mask_only: bool，是否只返回掩膜
    """
    class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 139]]
    if mean and std:
        for i in range(3):
            image[..., i] = image[..., i] * std[i]
            image[..., i] = image[..., i] + mean[i]
    mask = np.zeros([target.shape[0], target.shape[1], 3])
    for i in range(target.shape[2]):
        target_0 = target[..., i] * class_color[i][0]
        target_1 = target[..., i] * class_color[i][1]
        target_2 = target[..., i] * class_color[i][2]
        mask += np.stack([target_0, target_1, target_2], axis=2)
    image += np.uint8(mask)

    pair = {'mask': np.uint8(mask)}
    if not mask_only:
        pair['image'] = image

    return pair
