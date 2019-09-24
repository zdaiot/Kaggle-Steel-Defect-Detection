import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, pickle

class EncodedPixels2png():
    def __init__(self, csv_path, save_masks_path, save_mask_images):
        '''
        csv_path: 包含EncodedPixels的csv文件
        save_masks_path: 转换出来的masks存放路径
        save_mask_images: 是否将转换出来的masks文件保存下来
        '''
        self.csv_path = csv_path
        self.save_masks_path = save_masks_path
        self.save_mask_images = save_mask_images

        self.df = None
        self.deal_csv()

    # https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    def make_mask(self, row_id):
        '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
        fname = self.df.iloc[row_id].name
        labels = self.df.iloc[row_id][:4]
        masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
        # 4:class 1～4 (ch:0～3)

        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                mask = np.zeros(256 * 1600, dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos:(pos + le)] = 1
                masks[:, :, idx] = mask.reshape(256, 1600, order='F')
        return fname, masks

    def deal_csv(self):
        '''处理csv_path文件中的EncodedPixels
        '''
        df_raw = pd.read_csv(self.csv_path)
        # 将 ImageId 和 ClassId 分开；当ImageId='0002cc93b.jpg'的时候，ClassId有四个值。即每行为该图片属于某一类的的EncodedPixels
        df_raw['ImageId'], df_raw['ClassId'] = zip(*df_raw['ImageId_ClassId'].str.split('_'))
        df_raw['ClassId'] = df_raw['ClassId'].astype(int)
        # 声明新的 pandas 对象，每行为该图片属于所有类的的EncodedPixels（若有4个类标，则每行有五个值，分别为文件名、属于4类的EncodedPixels）
        self.df = df_raw.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
        # 增加一列 defeats，表示该图片 有多少个类别的EncodedPixels不为空
        self.df['defects'] = self.df.count(axis=1)

        if self.save_mask_images:
            if not os.path.exists(self.save_masks_path):
                os.makedirs(self.save_masks_path)

        masks_dict = {}
        for index in tqdm(range(len(self.df))):
            # 得到的 masks 维度为 [高，宽，类别数]
            fname, masks = self.make_mask(int(index))
            # print(fname, np.shape(masks), np.max(masks))s
            masks_dict[fname] = masks
        
        with open('filename.pickle', 'wb') as handle:
            pickle.dump(masks_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    EncodedPixels2png(csv_path='datasets/Steel_data/train.csv', save_masks_path='../Input/test_masks', save_mask_images=True)