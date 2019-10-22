import os
import cv2
import tqdm
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import codecs
import json
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from solver import Solver
from models.model import Model
from datasets.steel_dataset import provider
from utils.set_seed import seed_torch
from utils.cal_dice_iou import compute_dice_class
from config import get_seg_config


class ChooseThresholdMinArea():
    ''' 选择每一类的像素阈值和最小连通域
    '''
    def __init__(self, model, model_name, valid_loader, fold, save_path, class_num=4):
        ''' 模型初始化

        Args:
            model: 使用的模型
            model_name: 当前模型的名称
            valid_loader: 验证数据的Dataloader
            fold: 当前为多少折
            save_path: 保存结果的路径
            class_num: 有多少个类别
        '''
        self.model = model
        self.model_name = model_name
        self.valid_loader = valid_loader
        self.fold = fold
        self.save_path = save_path
        self.class_num = class_num

        self.model.eval()
        self.solver = Solver(model)

    def choose_threshold_minarea(self):
        ''' 采用网格法搜索各个类别最优像素阈值和最优最小连通域，并画出各个类别搜索过程中的热力图
        
        Return:
            best_thresholds_little: 每一个类别的最优阈值
            best_minareas_little: 每一个类别的最优最小连通取余
            max_dices_little: 每一个类别的最大dice值
        '''
        init_thresholds_range, init_minarea_range = np.arange(0.50, 0.71, 0.03), np.arange(768, 2305, 256)

        # 阈值列表和最小连通域列表，大小为 Nx4
        thresholds_table_big = np.array([init_thresholds_range, init_thresholds_range, \
                                         init_thresholds_range, init_thresholds_range])  # 阈值列表
        minareas_table_big = np.array([init_minarea_range, init_minarea_range, \
                                       init_minarea_range, init_minarea_range])  # 最小连通域列表

        f, axes = plt.subplots(figsize=(28.8, 18.4), nrows=2, ncols=self.class_num)
        cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)

        best_thresholds_big, best_minareas_big, max_dices_big = self.grid_search(thresholds_table_big, minareas_table_big, axes[0,:], cmap)
        print('best_thresholds_big:{}, best_minareas_big:{}, max_dices_big:{}'.format(best_thresholds_big, best_minareas_big, max_dices_big))

        # 开始细分类
        thresholds_table_little, minareas_table_little = list(), list()
        for best_threshold_big, best_minarea_big in zip(best_thresholds_big, best_minareas_big):
            thresholds_table_little.append(np.arange(best_threshold_big-0.03, best_threshold_big+0.03, 0.015))  # 阈值列表
            minareas_table_little.append(np.arange(best_minarea_big-256, best_minarea_big+257, 128))  # 像素阈值列表
        thresholds_table_little, minareas_table_little = np.array(thresholds_table_little), np.array(minareas_table_little)

        best_thresholds_little, best_minareas_little, max_dices_little = self.grid_search(thresholds_table_little, minareas_table_little, axes[1,:], cmap)
        print('best_thresholds_little:{}, best_minareas_little:{}, max_dices_little:{}'.format(best_thresholds_little, best_minareas_little, max_dices_little))

        f.savefig(os.path.join(self.save_path, self.model_name + '_fold'+str(self.fold)), bbox_inches='tight')
        # plt.show()
        plt.close()

        return best_thresholds_little, [float(x) for x in best_minareas_little], max_dices_little

    def grid_search(self, thresholds_table, minareas_table, axes, cmap):
        ''' 给定包含各个类别搜索区间的thresholds_table和minareas_table，求的各个类别的最优像素阈值，最优最小连通域，最高dice；
            并画出各个类别搜索过程中的热力图

        Args:
            thresholds_table: 待搜索的阈值范围，维度为[4, N]，numpy类型
            minareas_table: 待搜索的最小连通域范围，维度为[4, N]，numpy类型
            axes: 画各个类别搜索热力图时所需要的画柄，尺寸为[class_num]
            cmap: 画图时所需要的cmap

        return:
            best_thresholds: 各个类别的最优像素阈值，尺寸为[class_num]
            best_minareas: 各个类别的最优最小连通域，尺寸为[class_num]
            max_dices: 各个类别的最大dice，尺寸为[class_num]
        '''
        dices_table = np.zeros((self.class_num, np.shape(thresholds_table)[1], np.shape(minareas_table)[1]))
        tbar = tqdm.tqdm(self.valid_loader)
        with torch.no_grad():
            for i, samples in enumerate(tbar):
                if len(samples) == 0:
                    continue
                images, masks = samples[0], samples[1]
                # 完成网络的前向传播
                masks_predict_allclasses = self.solver.forward(images)
                dices_table += self.grid_search_batch(thresholds_table, minareas_table, masks_predict_allclasses, masks)

        dices_table = dices_table/len(tbar)
        best_thresholds, best_minareas, max_dices = list(), list(), list()
        # 处理每一类的预测结果
        for each_class, dices_oneclass_table in enumerate(dices_table):
            max_dice = np.max(dices_oneclass_table)
            max_location = np.unravel_index(np.argmax(dices_oneclass_table, axis=None),
                                            dices_oneclass_table.shape)
            best_thresholds.append(thresholds_table[each_class, max_location[0]])
            best_minareas.append(minareas_table[each_class, max_location[1]])
            max_dices.append(max_dice)

            data = pd.DataFrame(data=dices_oneclass_table, index=np.around(thresholds_table[each_class,:], 3), columns=minareas_table[each_class,:])
            sns.heatmap(data, linewidths=0.05, ax=axes[each_class], vmax=np.max(dices_oneclass_table), vmin=np.min(dices_oneclass_table), cmap=cmap,
                        annot=True, fmt='.4f')
            axes[each_class].set_title('search result')
        return best_thresholds, best_minareas, max_dices

    def grid_search_batch(self, thresholds_table, minareas_table, masks_predict_allclasses, masks_allclasses):
        '''给定thresholds、minareas矩阵、一个batch的预测结果和真实标签，遍历每个类的每一个组合得到对应的dice值

        Args:
            thresholds_table: 待搜索的阈值范围，维度为[4, N]，numpy类型
            minareas_table: 待搜索的最小连通域范围，维度为[4, N]，numpy类型
            masks_predict_allclasses: 所有类别的某个batch的预测结果且未经过sigmoid，维度为[batch_size, class_num, height, width]
            masks_allclasses: 所有类别的某个batch的真实类标，维度为[batch_size, class_num, height, width]

        Return:
            dices_table: 各个类别在其各自的所有搜索组合中所得到的dice值，维度为[4, M, N]
        '''

        # 得到每一个类别的搜索阈值区间和最小连通域搜索区间
        dices_table = list()
        for each_class, (thresholds_range, minareas_range) in enumerate(zip(thresholds_table, minareas_table)):
            # 得到每一类的预测结果和真实类标
            masks_predict_oneclass = masks_predict_allclasses[:, each_class, ...]
            masks_oneclasses = masks_allclasses[:, each_class, ...]

            dices_range = self.post_process(thresholds_range, minareas_range, masks_predict_oneclass, masks_oneclasses)
            dices_table.append(dices_range)
        # 得到的大小为 4 x len(thresholds_range) x len(minareas_range)
        return np.array(dices_table)

    def post_process(self, thresholds_range, minareas_range, masks_predict_oneclass, masks_oneclasses):
        '''给定某个类别的某个batch的数据，遍历所有搜索组合，得到每个组合的dice值
        
        Args:
            thresholds_range: 具体某个类别的像素阈值搜索区间，尺寸为[M]
            minareas_range: 具体某个类别的最小连通域搜索区间，尺寸为[N]
            masks_predict_oneclass: 预测出的某个类别的该batch的tensor向量且未经过sigmoid，维度为[batch_size, height, width]
            masks_oneclasses: 某个类别的该batch的真实类标，维度为[batch_size, height, width]
        
        Return:
            dices_range: 某个类别的该batch的所有搜索组合得到dice矩阵，维度为[M, N]
        '''

        # 注意，损失函数中包含sigmoid函数，一般情况下需要手动经过sigmoid函数
        masks_predict_oneclass = torch.sigmoid(masks_predict_oneclass).detach().cpu().numpy()
        dices_range = np.zeros((len(thresholds_range), len(minareas_range)))

        # 遍历每一个像素阈值和最小连通域
        for index_threshold, threshold in enumerate(thresholds_range):
            for index_minarea, minarea in enumerate(minareas_range):
                batch_preds = list()
                # 遍历每一张图片
                for pred in masks_predict_oneclass:
                    mask = cv2.threshold(pred, threshold, 1, cv2.THRESH_BINARY)[1]
                    # 将背景标记为 0，其他的块从 1 开始的正整数标记
                    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
                    predictions = np.zeros((256, 1600), np.float32)
                    num = 0
                    for c in range(1, num_component):
                        p = (component == c)
                        if p.sum() > minarea:
                            predictions[p] = 1
                            num += 1
                    batch_preds.append(predictions)
                dice = compute_dice_class(torch.from_numpy(np.array(batch_preds)), masks_oneclasses)
                dices_range[index_threshold, index_minarea] = dice
        return dices_range


def get_model(model_name, load_path):
    ''' 加载网络模型并加载对应的权重

    Args: 
        model_name: 当前模型的名称
        load_path: 当前模型的权重路径
    
    Return:
        model: 加载出来的模型
    '''
    model = Model(model_name).create_model()
    Solver(model).load_checkpoint(load_path)
    return model


if __name__ == "__main__":
    config = get_seg_config()
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    mask_only = True

    dataloaders = provider(config.dataset_root, os.path.join(config.dataset_root, 'train.csv'), mean, std, config.batch_size, config.num_workers, config.n_splits, mask_only)
    results = {}
    # 存放权重的路径
    model_path = os.path.join(config.save_path, config.model_name)
    best_thresholds_sum, best_minareas_sum, max_dices_sum = [0 for x in range(len(dataloaders))], \
                                                            [0 for x in range(len(dataloaders))], [0 for x in range(len(dataloaders))]
    for fold_index, [train_loader, valid_loader] in enumerate(dataloaders):
        if fold_index != 1:
            continue
        # 存放权重的路径+文件名
        load_path = os.path.join(model_path, '%s_fold%d_best.pth' % (config.model_name, fold_index))
        # 加载模型
        model = get_model(config.model_name, load_path)
        mychoose_threshold_minarea = ChooseThresholdMinArea(model, config.model_name, valid_loader, fold_index, model_path)
        best_thresholds, best_minareas, max_dices = mychoose_threshold_minarea.choose_threshold_minarea()
        result = {'best_thresholds': best_thresholds, 'best_minareas': best_minareas, 'max_dices': max_dices}
        results[str(fold_index)] = result

        best_thresholds_sum = [x+y for x,y in zip(best_thresholds_sum, best_thresholds)]
        best_minareas_sum = [x+y for x,y in zip(best_minareas_sum, best_minareas)]
        max_dices_sum = [x+y for x,y in zip(max_dices_sum, max_dices)]
    best_thresholds_average, best_minareas_average, max_dices_average = [x/len(dataloaders) for x in best_thresholds_sum], \
                                                                        [x/len(dataloaders) for x in best_minareas_sum], [x/len(dataloaders) for x in max_dices_sum]
    results['mean'] = {'best_thresholds': best_thresholds_average, 'best_minareas': best_minareas_average, 'max_dices': max_dices_average}
    with codecs.open(model_path + '/%s_result.json' % (config.model_name), 'w', "utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False)

    print('save the result')
