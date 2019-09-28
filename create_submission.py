import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

kaggle = 0
if kaggle:
    os.system('pip install /kaggle/input/segmentation-models/pretrainedmodels-0.7.4/ > /dev/null')
    os.system('pip install /kaggle/input/segmentation-models/segmentation_models.pytorch/ > /dev/null')
    package_path = 'kaggle/input/sources' # add unet script dataset
    import sys
    sys.path.append(package_path)
from datasets.steel_dataset import TestDataset
from classify_segment import Classify_Segment_Folds, Classify_Segment_Fold


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


def create_submission(n_splits, model_name, batch_size, num_workers, mean, std, test_data_folder, sample_submission_path, model_path):
    '''

    :param n_splits: 折数，类型为list
    :param model_name: 当前模型的名称
    :param batch_size: batch的大小
    :param num_workers: 加载数据的线程
    :param mean: 均值
    :param std: 方差
    :param test_data_folder: 测试数据存放的路径
    :param sample_submission_path: 提交样例csv存放的路径
    :param model_path: 当前模型权重存放的目录
    :return: None
    '''
    # 加载数据集
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(test_data_folder, df, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    if len(n_splits) == 1:
        classify_segment = Classify_Segment_Fold(model_name, n_splits[0], model_path).classify_segment
    else:
        classify_segment = Classify_Segment_Folds(model_name, n_splits, model_path, testset).classify_segment_folds

    # start prediction
    predictions = []
    for i, (fnames, images) in enumerate(tqdm(testset)):
        results = classify_segment(images).detach().cpu().numpy()

        for fname, preds in zip(fnames, results):
            for cls, pred in enumerate(preds):
                rle = mask2rle(pred)
                name = fname + str(cls+1)
                predictions.append([name, rle])

    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    # 设置超参数
    model_name = 'unet_resnet34'
    num_workers = 12
    batch_size = 8
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    n_splits = [1] # [0, 1, 2, 3, 4]

    if kaggle:
        sample_submission_path = 'kaggel/input/severstal-steel-defect-detection/sample_submission.csv'
        test_data_folder = "kaggle/input/severstal-steel-defect-detection/test_images"
        model_path = 'kaggle/input/checkpoints'
    else:
        sample_submission_path = 'datasets/Steel_data/sample_submission.csv'
        test_data_folder = 'datasets/Steel_data/test_images'
        model_path = './checkpoints/' + model_name

    create_submission(n_splits, model_name, batch_size, num_workers, mean, std, test_data_folder,
                      sample_submission_path, model_path)
