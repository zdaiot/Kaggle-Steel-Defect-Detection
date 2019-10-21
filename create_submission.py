import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

kaggle = 0
if kaggle:
    os.system('pip install /kaggle/input/segmentation_models/pretrainedmodels-0.7.4/ > /dev/null')
    os.system('pip install /kaggle/input/segmentation_models/EfficientNet-PyTorch/ > /dev/null')
    os.system('pip install /kaggle/input/segmentation_models/segmentation_models.pytorch/ > /dev/null')
    package_path = '/kaggle/input/sources' # add unet script dataset
    import sys
    sys.path.append(package_path)
from classify_segment import Classify_Segment_Folds, Classify_Segment_Fold, Classify_Segment_Folds_Split


class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image)
        return fname, images

    def __len__(self):
        return self.num_samples

# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_submission(classify_splits, seg_splits, batch_size, num_workers, mean, std, test_data_folder, sample_submission_path, model_path, tta_flag=False, average_strategy=False, kaggle=0):
    '''

    :param classify_splits: 分类模型的折数，类型为字典
    :param seg_splits: 分割模型的折数，类型为字典
    :param batch_size: batch的大小
    :param num_workers: 加载数据的线程
    :param mean: 均值
    :param std: 方差
    :param test_data_folder: 测试数据存放的路径
    :param sample_submission_path: 提交样例csv存放的路径
    :param model_path: 当前模型权重存放的目录
    :param tta_flag: 是否使用tta
    :param average_strategy: 是否使用平均策略
    :param kaggle: 线上或线下
    :return: None
    '''
    # 加载数据集
    df = pd.read_csv(sample_submission_path)
    test_loader = DataLoader(
        TestDataset(test_data_folder, df, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    if len(classify_splits) == 1 and len(seg_splits) == 1:
        classify_segment = Classify_Segment_Fold(classify_splits, seg_splits, model_path, tta_flag=tta_flag, kaggle=kaggle).classify_segment
    else:
        classify_segment = Classify_Segment_Folds_Split(classify_splits, seg_splits, model_path, tta_flag=tta_flag, kaggle=kaggle).classify_segment_folds

    # start prediction
    predictions = []
    for i, (fnames, images) in enumerate(tqdm(test_loader)):
        if len(classify_splits) == 1 and len(seg_splits) == 1:
            results = classify_segment(images).detach().cpu().numpy()
        else:
            results = classify_segment(images, average_strategy=average_strategy).detach().cpu().numpy()

        for fname, preds in zip(fnames, results):
            for cls, pred in enumerate(preds):
                rle = mask2rle(pred)
                name = fname + '_' + str(cls+1)
                predictions.append([name, rle])

    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    # 设置超参数
    num_workers = 12
    batch_size = 1
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    classify_splits = {'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1} # 'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1
    segment_splits = {'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1} # 'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1
    tta_flag = True
    average_strategy = False

    if kaggle:
        sample_submission_path = '/kaggle/input/severstal-steel-defect-detection/sample_submission.csv'
        test_data_folder = "/kaggle/input/severstal-steel-defect-detection/test_images"
        model_path = '/kaggle/input/checkpoints'
    else:
        sample_submission_path = 'datasets/Steel_data/sample_submission.csv'
        test_data_folder = 'datasets/Steel_data/test_images'
        model_path = './checkpoints/'

    create_submission(classify_splits, segment_splits, batch_size, num_workers, mean, std, test_data_folder,
                      sample_submission_path, model_path, tta_flag=tta_flag, average_strategy=average_strategy, kaggle=kaggle)
