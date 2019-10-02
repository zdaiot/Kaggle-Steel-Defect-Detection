import torch
from torch import nn
from albumentations import HorizontalFlip
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

from datasets.steel_dataset import TestDataset, classify_provider
from models.model import ClassifyResNet


class ClassifyTest():
    def __init__(self, model, threshold=[0.5, 0.5, 0.5, 0.5], tta=False):
        self.threshold = threshold
        self.tta = tta

        self.model = model
        self.model.eval()

    def predict_dataloader(self, dataloader):
        """对测试集进行测试

        Return:
            test_image_id: 测试样本的名称
            predict_label: 各个样本对应的预测类标
        """
        tbar = tqdm(dataloader)
        test_probility = list()
        test_image_id = list()
        for (fnames, images) in tbar:
            images = images.cuda()
            probility = self.tta_pred(images)
            probility = probility.data.cpu().numpy()
            test_probility.append(probility)
            test_image_id.extend([fname for fname in fnames])
        test_probility = np.concatenate(test_probility)
        predict_label = test_probility > np.array(self.threshold).reshape(1, 4, 1, 1)
        
        return test_image_id, predict_label

    def predict_image(self, images):
        """对一个batch的样本进行测试

        Return:
            predict_label: 各个样本对应的预测类标
        """
        probility = self.tta_pred(images)
        probility = probility.data.cpu().numpy()
        predict_label = probility > np.array(self.threshold).reshape(1, 4, 1, 1)

        return predict_label

    def tta_pred(self, images):
        # 水平翻转
        probility_tta = 0
        logit = self.model(torch.flip(images, dims=[3]))
        probility = torch.sigmoid(logit)
        probility_tta += probility

        # 原始
        logit = self.model(images)
        probility = torch.sigmoid(logit)
        probility_tta += probility

        probility_tta /= 2

        return probility_tta


if __name__ == "__main__":
    data_folder = "/home/apple/program/MXQ/Competition/Kaggle/Steal-Defect/Kaggle-Steel-Defect-Detection/datasets/Steel_data"
    df_path = "/home/apple/program/MXQ/Competition/Kaggle/Steal-Defect/Kaggle-Steel-Defect-Detection/datasets/Steel_data/train.csv"
    test_df = pd.read_csv('./datasets/Steel_data/sample_submission.csv')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_dataset = TestDataset('./datasets/Steel_data/test_images', test_df, mean, std)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    model = ClassifyResNet('unet_resnet34', 4, training=False)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    pth_path = "checkpoints/unet_resnet34/unet_resnet34_classify_fold1.pth"
    checkpoint = torch.load(pth_path)
    model.module.load_state_dict(checkpoint['state_dict'])

    class_test = ClassifyTest(model, [0.5, 0.5, 0.5, 0.5], True)
    # 直接对一整个数据集进行预测
    # image_id, predict_label = class_test.predict(dataloader)
    # 按照mini-batch的方式进行预测
    class_dataloader = classify_provider(data_folder, df_path, mean, std, 20, 8, 5)
    for fold_index, [train_dataloader, val_dataloader] in enumerate(class_dataloader):
        train_bar = tqdm(val_dataloader)
        class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (139, 0, 139)]
        number_sample = 0
        num_true = 0
        for (images, targets) in train_bar:
            images = images.cuda()
            # 预测并计算指标
            predicts = class_test.predict_image(images).squeeze().astype(int)
            targets_numpy = targets.data.cpu().numpy()
            num_true += (predicts == targets_numpy).sum()
            number_sample += targets_numpy.size
            descript = 'True / Num: %d / %d' % (num_true, number_sample)
            train_bar.set_description(desc=descript)

            image = images[0]
            for i in range(3):
                image[i] = image[i] * std[i]
                image[i] = image[i] + mean[i]
            image = image.permute(1, 2, 0).cpu().numpy()
            target = targets[0]
            # 真实类别标签
            position_x = 10
            for i in range(target.size(0)):
                color = class_color[i]
                position_x += 50
                position = (position_x, 50)
                if target[i] != 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    image = cv2.putText(image, str(i), position, font, 1.2, color, 2)
            # 预测类别标签
            predict = predicts[0]
            position_x = 10
            for i in range(predict.shape[0]):
                color = class_color[i]
                position_x += 50
                position = (position_x, 100)
                if predict[i] != 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    image = cv2.putText(image, str(i), position, font, 1.2, color, 2)
            cv2.imshow('win', image)
            cv2.waitKey(240)
        print("Accuracy: %.4f" % (num_true / number_sample))
    pass
