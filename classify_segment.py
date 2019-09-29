import json
import os
import cv2
import torch
import numpy as np
from solver import Solver
from models.model import Model, ClassifyResNet


class Get_Classify_Results():
    def __init__(self, model_name, fold, model_path, class_num=4):
        ''' 处理当前fold一个batch的数据分类结果

        :param model_name: 当前的模型名称
        :param fold: 当前的折数
        :param model_path: 存放所有模型的路径
        :param class_num: 类别总数
        '''
        self.model_name = model_name
        self.fold = fold
        self.model_path = model_path
        self.class_num = class_num

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型及其权重
        self.classify_model = ClassifyResNet(model_name)
        if torch.cuda.is_available():
            self.classify_model = torch.nn.DataParallel(self.classify_model)

        self.classify_model.to(self.device)

        self.classify_model_path = os.path.join(self.model_path, '%s_classify_fold%d_best.pth' % (self.model_name, self.fold))
        self.solver = Solver(self.classify_model)
        self.classify_model = self.solver.load_checkpoint(self.classify_model_path)
        self.classify_model.eval()

    def get_classify_results(self, images, thrshold=0.5):
        ''' 处理当前fold一个batch的数据分类结果

        :param images: 一个batch的数据，维度为[batch, channels, height, width]
        :param thrshold: 分类模型的阈值
        :return: predict_classes: 一个batch的数据经过分类模型后的结果，维度为[batch, class_num]
        '''
        predict_classes = self.solver.forward(images)
        predict_classes = predict_classes > thrshold
        return predict_classes


class Get_Segment_Results():
    def __init__(self, model_name, fold, model_path, class_num=4):
        ''' 处理当前fold一个batch的数据分割结果

        :param model_name: 当前的模型名称
        :param fold: 当前的折数
        :param model_path: 存放所有模型的路径
        :param class_num: 类别总数
        '''
        self.model_name = model_name
        self.fold = fold
        self.model_path = model_path
        self.class_num = class_num

        # 加载模型及其权重
        self.segment_model = Model(self.model_name).create_model()
        self.segment_model_path = os.path.join(self.model_path, '%s_fold%d_best.pth' % (self.model_name, self.fold))
        self.solver = Solver(self.segment_model)
        self.segment_model = self.solver.load_checkpoint(self.segment_model_path)
        self.segment_model.eval()

        # 加载存放像素阈值和连通域的json文件
        self.json_path = os.path.join(self.model_path, 'result.json')
        self.best_thresholds, self.best_minareas = self.get_thresholds_minareas(self.json_path, self.fold)

    def get_segment_results(self, images):
        ''' 处理当前fold一个batch的数据分割结果

        :param images: 一个batch的数据，维度为[batch, channels, height, width]
        :return: predict_masks: 一个batch的数据经过分割网络后得到的预测结果，维度为[batch, class_num, height, width]
        '''
        predict_masks = self.solver.forward(images)
        for index, predict_masks_classes in enumerate(predict_masks):
            for each_class, pred in enumerate(predict_masks_classes):
                pred_binary, _ = self.post_process(pred.detach().cpu().numpy(), self.best_thresholds[each_class], self.best_minareas[each_class])
                predict_masks[index, each_class] = torch.from_numpy(pred_binary)
        return predict_masks

    def post_process(self, probability, threshold, min_size):
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

    def get_thresholds_minareas(self, json_path, fold):
        ''' 得到各个类别的特定fold的最优像素阈值和最优最小连通域

        :param json_path: 要加载的json路径
        :param fold: 要加载哪一折的结果
        :return: best_thresholds: 各个类别的最优像素阈值，类型为list
        :return: best_minareas: 各个类别的最优最小连通域，类型为list
        '''
        with open(json_path, encoding='utf-8') as json_file:
            result = json.load(json_file)
        best_thresholds, best_minareas = result[str(fold)]['best_thresholds'], result[str(fold)]['best_minareas']
        return best_thresholds, best_minareas


class Classify_Segment_Fold():
    def __init__(self, model_name, fold, model_path, class_num=4):
        ''' 处理当前fold一个batch的分割结果和分类结果

        :param model_name: 当前的模型名称
        :param fold: 当前的折数
        :param model_path: 存放所有模型的路径
        :param class_num: 类别总数
        '''
        self.model_name = model_name
        self.fold = fold
        self.model_path = model_path
        self.class_num = class_num

        self.classify_model = Get_Classify_Results(self.model_name, self.fold, self.model_path, self.class_num)
        self.segment_model = Get_Segment_Results(self.model_name, self.fold, self.model_path, self.class_num)

    def classify_segment(self, images):
        ''' 处理当前fold一个batch的分割结果和分类结果

        :param images: 一个batch的数据，维度为[batch, channels, height, width]
        :return: predict_masks，一个batch的数据，经过分割模型和分类模型处理后的结果
        '''
        # 得到一个batch数据分类模型的结果，维度为[batch, class_num]
        predict_classes = self.classify_model.get_classify_results(images)
        # 得到一个batch数据分割模型的结果，维度为[batch, class_num, height, width]
        predict_masks = self.segment_model.get_segment_results(images)
        for index, predicts in enumerate(predict_classes):
            for each_class, pred in enumerate(predicts):
                if pred == 0:
                    predict_masks[index, each_class, ...] = 0
        return predict_masks


class Classify_Segment_Folds():
    def __init__(self, model_name, n_splits, model_path, class_num=4):
        ''' 使用投票法处理所有fold一个batch的分割结果和分类结果

        :param model_name: 当前的模型名称
        :param n_splits: 总共有多少折，为list列表
        :param model_path: 存放所有模型的路径
        :param class_num: 类别总数
        '''
        self.model_name = model_name
        self.n_splits = n_splits
        self.model_path = model_path
        self.class_num = class_num

        self.classify_models, self.segment_models = list(), list()
        self.get_classify_segment_models()

    def get_classify_segment_models(self):
        ''' 加载所有折的分割模型和分类模型
        '''

        for fold in self.n_splits:
            self.classify_models.append(Get_Classify_Results(self.model_name, fold, self.model_path, self.class_num))
            self.segment_models.append(Get_Segment_Results(self.model_name, fold, self.model_path, self.class_num))

    def classify_segment_folds(self, images):
        ''' 使用投票法处理所有fold一个batch的分割结果和分类结果

        :param images: 一个batch的数据，维度为[batch, channels, height, width]
        :return: results，使用投票法处理所有fold一个batch的分割结果和分类结果，维度为[batch, class_num, height, width]
        '''
        results = torch.zeros(images.shape[0], self.class_num, images.shape[2], images.shape[3])
        for classify_model, segment_model in zip(self.classify_models, self.segment_models):
            # 得到一个batch数据分类模型的结果，维度为[batch, class_num]
            predict_classes = classify_model.get_classify_results(images)
            # 得到一个batch数据分割模型的结果，维度为[batch, class_num, height, width]
            predict_masks = segment_model.get_segment_results(images)
            for index, predicts in enumerate(predict_classes):
                for each_class, pred in enumerate(predicts):
                    if pred == 0:
                        predict_masks[index, each_class, ...] = 0
            results += predict_masks.detach().cpu()
        vote_model_num = len(self.n_splits)
        vote_ticket = round(vote_model_num / 2.0)
        results = results > vote_ticket

        return results


class Segment_Folds():
    def __init__(self, model_name, n_splits, model_path, class_num=4):
        ''' 使用投票法处理所有fold一个batch的分割结果

        :param model_name: 当前的模型名称
        :param n_splits: 总共有多少折，为list列表
        :param model_path: 存放所有模型的路径
        :param class_num: 类别总数
        '''
        self.model_name = model_name
        self.n_splits = n_splits
        self.model_path = model_path
        self.class_num = class_num

        self.segment_models = list()
        self.get_segment_models()

    def get_segment_models(self):
        ''' 加载所有折的分割模型
        '''

        for fold in self.n_splits:
            self.segment_models.append(Get_Segment_Results(self.model_name, fold, self.model_path, self.class_num))

    def segment_folds(self, images):
        ''' 使用投票法处理所有fold一个batch的分割结果

        :param images: 一个batch的数据，维度为[batch, channels, height, width]
        :return: results，使用投票法处理所有fold一个batch的分割结果和分类结果，维度为[batch, class_num, height, width]
        '''
        results = torch.zeros(images.shape[0], self.class_num, images.shape[2], images.shape[3])
        for segment_model in self.segment_models:
            # 得到一个batch数据分割模型的结果，维度为[batch, class_num, height, width]
            predict_masks = segment_model.get_segment_results(images)
            results += predict_masks.detach().cpu()
        vote_model_num = len(self.n_splits)
        vote_ticket = round(vote_model_num / 2.0)
        results = results > vote_ticket

        return results