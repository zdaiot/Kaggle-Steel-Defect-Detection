kaggle = 0
import os
if kaggle:
    os.system('pip install /kaggle/input/segmentation-models/pretrainedmodels-0.7.4/ > /dev/null')
    os.system('pip install /kaggle/input/segmentation-models/segmentation_models.pytorch/ > /dev/null')
    package_path = '../input/models' # add unet script dataset
    import sys
    sys.path.append(package_path)
    from model import Model
else:
    from models.model import Model
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensor
from datasets.steel_dataset import TestDataset


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


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def get_model(model_name, ckpt_path):
    # 加载模型
    model = Model(model_name, encoder_weights=None).create_model()
    # Initialize mode and load trained weights

    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.module.load_state_dict(state["state_dict"])

    return model


def create_submission(best_threshold, min_size, batch_size, num_workers, mean, std, test_data_folder, sample_submission_path, model):
    # 加载数据集
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(test_data_folder, df, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # start prediction
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        fnames, images = batch
        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()
        for fname, preds in zip(fnames, batch_preds):
            for cls, pred in enumerate(preds):
                pred, num = post_process(pred, best_threshold, min_size)
                rle = mask2rle(pred)
                name = fname + f"_{cls+1}"
                predictions.append([name, rle])

    # save predictions to submission.csv
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    if kaggle:
        sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
        test_data_folder = "../input/severstal-steel-defect-detection/test_images"
        ckpt_path = "../input/checkpoints/unet_resnet34_fold0_best.pth"
    else:
        sample_submission_path = 'datasets/Steel_data/sample_submission.csv'
        test_data_folder = 'datasets/Steel_data/test_images'
        ckpt_path = 'checkpoints/unet_resnet34/unet_resnet34_fold0_best.pth'

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
    create_submission(best_threshold, min_size, batch_size, num_workers, mean, std, test_data_folder,
                      sample_submission_path, model)
