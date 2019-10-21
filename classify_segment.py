import json
import os
import cv2
import torch
import numpy as np
from solver import Solver
from models.model import Model, ClassifyResNet


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


def get_thresholds_minareas(json_path, fold=None):
    ''' 得到各个类别的特定fold的最优像素阈值和最优最小连通域或者所有fold的平均最优像素阈值和平均最优最小连通域

    :param json_path: 要加载的json路径
    :param fold: 要加载哪一折的结果，当fold为None的时候，返回的是平均值
    :return: thresholds: 当fold为非None的时候，返回特定fold各个类别的最优像素阈值，类型为list；
                         当fold为None的时候，返回所有fold各个类别的最优像素阈值的平均值，类型为list
    :return: minareas: 当fold为非None的时候，返回特定fold各个类别的最优最小连通域，类型为list
                         当fold为None的时候，返回所有fold各个类别的最优最小连通域的平均值，类型为list
    '''
    with open(json_path, encoding='utf-8') as json_file:
        result = json.load(json_file)
    if fold != None:
        thresholds, minareas = result[str(fold)]['best_thresholds'], result[str(fold)]['best_minareas']
    else:
        thresholds, minareas = result['mean']['best_thresholds'], result['mean']['best_minareas']
    return thresholds, minareas


class Get_Classify_Results():
    def __init__(self, model_name, fold, model_path, class_num=4, tta_flag=False):
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
        self.tta_flag = tta_flag
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型及其权重
        self.classify_model = ClassifyResNet(model_name, encoder_weights=None)
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
        if self.tta_flag:
            predict_classes = self.solver.tta(images, seg=False)
        else:
            predict_classes = self.solver.forward(images)
        predict_classes = predict_classes > thrshold
        return predict_classes


class Get_Segment_Results():
    def __init__(self, model_name, fold, model_path, class_num=4, tta_flag=False):
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
        self.tta_flag = tta_flag

        # 加载模型及其权重
        self.segment_model = Model(self.model_name, encoder_weights=None).create_model()
        self.segment_model_path = os.path.join(self.model_path, '%s_fold%d_best.pth' % (self.model_name, self.fold))
        self.solver = Solver(self.segment_model)
        self.segment_model = self.solver.load_checkpoint(self.segment_model_path)
        self.segment_model.eval()

        # 加载存放像素阈值和连通域的json文件
        self.json_path = os.path.join(self.model_path, '%s_result.json' % self.model_name)
        self.best_thresholds, self.best_minareas = get_thresholds_minareas(self.json_path, self.fold)

    def get_segment_results(self, images, process_flag=True):
        ''' 处理当前fold一个batch的数据分割结果

        :param images: 一个batch的数据，维度为[batch, channels, height, width]
        :param process_flag: 是否经过像素阈值和最小连通域
        :return: predict_masks: 一个batch的数据经过分割网络后得到的预测结果，维度为[batch, class_num, height, width]。
            当 process_flag=True 的时候，返回的结果经过了阈值以及最小连通域，得到的 predict_masks 为二值化的
            当 process_flag=False 的时候，返回的结果未经过阈值以及最小连通域，得到的 predict_masks 为非二值化的，值处于 [0, 1] 之间
        '''
        # 得到的维度为[batch, class_num, height, width]
        if self.tta_flag:
            predict_masks = self.solver.tta(images)
        else:
            predict_masks = self.solver.forward(images)
        # 是否需要经过阈值以及像素阈值，默认经过
        if process_flag:
            for index, predict_masks_classes in enumerate(predict_masks):
                for each_class, pred in enumerate(predict_masks_classes):
                    pred_binary, _ = post_process(pred.detach().cpu().numpy(), self.best_thresholds[each_class], self.best_minareas[each_class])
                    predict_masks[index, each_class] = torch.from_numpy(pred_binary)
        return predict_masks


class Classify_Segment_Fold():
    def __init__(self, classify_fold, seg_fold, model_path, class_num=4, tta_flag=False, kaggle=0):
        ''' 处理当前fold一个batch的分割结果和分类结果

        :param model_name: 当前的模型名称
        :param classify_fold: 字典，分类模型 {'model_name': fold_index}
        :param seg_fold: 字典，分割模型 {'model_name': fold_index}
        :param model_path: 存放所有模型的路径
        :param class_num: 类别总数
        '''
        self.classify_fold = classify_fold
        self.seg_fold = seg_fold
        self.model_path = model_path
        self.class_num = class_num
        for (model_name, fold) in self.classify_fold.items():
            if kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:
                pth_path = self.model_path                
            self.classify_model = Get_Classify_Results(model_name, fold, pth_path, self.class_num, tta_flag=tta_flag)
        for (model_name, fold) in self.classify_fold.items():
            if kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:
                pth_path = self.model_path
            self.segment_model = Get_Segment_Results(model_name, fold, pth_path, self.class_num, tta_flag=tta_flag)

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
    def __init__(self, classify_folds, segment_folds, model_path, class_num=4, tta_flag=False, kaggle=0):
        ''' 使用投票法处理所有fold一个batch的分割结果和分类结果

        :param classify_folds: 字典，{'model_name': fold_index}
        :param segment_folds: 字典，{'model_name': fold_index}
        :param model_path: 存放所有模型的路径
        :param class_num: 类别总数
        :param kaggle: 是否在kaggle的kernel运行
        '''
        self.classify_folds = classify_folds
        self.segment_folds = segment_folds
        self.model_path = model_path
        self.class_num = class_num
        self.tta_flag = tta_flag
        self.kaggle = kaggle

        self.classify_models, self.segment_models = list(), list()
        self.get_classify_segment_models()

    def get_classify_segment_models(self):
        ''' 加载所有折的分割模型和分类模型
        '''
        for (model_name, fold) in self.classify_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path                
            self.classify_models.append(Get_Classify_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))
        for (model_name, fold) in self.segment_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path
            self.segment_models.append(Get_Segment_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))

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
        vote_model_num = len(self.segment_folds)
        vote_ticket = round(vote_model_num / 2.0)
        results = results > vote_ticket

        return results


class Classify_Segment_Folds_Split():
    def __init__(self, classify_folds, segment_folds, model_path, class_num=4, tta_flag=False, kaggle=0):
        ''' 首先得到分类模型的集成结果，再得到分割模型的集成结果，最后将两个结果进行融合

        :param classify_folds: 字典，{'model_name': fold_index}
        :param segment_folds: 字典，{'model_name': fold_index}
        :param model_path: 存放所有模型的路径, checkpoints/
        :param class_num: 类别总数
        :param kaggle: 是否在kaggle的kernel运行
        '''
        self.classify_folds = classify_folds
        self.segment_folds = segment_folds
        self.model_path = model_path
        self.class_num = class_num
        self.tta_flag = tta_flag
        self.kaggle = kaggle

        self.classify_models, self.segment_models = list(), list()
        self.get_classify_segment_models()

    def get_classify_segment_models(self):
        ''' 加载所有折的分割模型和分类模型
        '''
        for (model_name, fold) in self.classify_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path                
            self.classify_models.append(Get_Classify_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))
        for (model_name, fold) in self.segment_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path
            self.segment_models.append(Get_Segment_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))

    def classify_segment_folds(self, images, average_strategy=False):
        ''' 使用投票法或者平均法处理所有fold一个batch的分割结果和分类结果

        :param images: 一个batch的数据，维度为[batch, channels, height, width]
        :param average_strategy: 当为True的时候，使用平均策略；当为False的时候，使用投票策略
        :return: results，使用投票法或者平均法处理所有fold一个batch的分割结果和分类结果，维度为[batch, class_num, height, width]
        '''
        classify_results = torch.zeros(images.shape[0], self.class_num)
        segment_results = torch.zeros(images.shape[0], self.class_num, images.shape[2], images.shape[3])
        # 得到分类结果
        for classify_index, classify_model in enumerate(self.classify_models):
            classify_result_fold = classify_model.get_classify_results(images)
            classify_results += classify_result_fold.detach().cpu().squeeze().float()
        classify_vote_model_num = len(self.classify_folds)
        classify_vote_ticket = round(classify_vote_model_num / 2.0)
        classify_results = classify_results > classify_vote_ticket

        # 得到分割结果
        # 如果采用平均策略的话
        if average_strategy:
            for segment_index, segment_model in enumerate(self.segment_models):
                segment_result_fold = segment_model.get_segment_results(images, process_flag=False)
                segment_results += segment_result_fold.detach().cpu()
            average_thresholds, average_minareas = get_thresholds_minareas(os.path.join(self.model_path, 'result.json'))
            segment_results = segment_results/len(self.segment_folds)
            for index, predict_masks_classes in enumerate(segment_results):
                for each_class, pred in enumerate(predict_masks_classes):
                    pred_binary, _ = post_process(pred.detach().cpu().numpy(), average_thresholds[each_class], average_minareas[each_class])
                    segment_results[index, each_class] = torch.from_numpy(pred_binary)
        # 如果采用投票策略的话
        else:
            for segment_index, segment_model in enumerate(self.segment_models):
                segment_result_fold = segment_model.get_segment_results(images)
                segment_results += segment_result_fold.detach().cpu()
            segment_vote_model_num = len(self.segment_folds)
            segment_vote_ticket = round(segment_vote_model_num / 2.0)
            segment_results = segment_results > segment_vote_ticket

        # 将分类结果和分割结果进行融合
        for batch_index, classify_result in enumerate(classify_results):
            segment_results[batch_index, ~classify_result, ...] = 0

        return segment_results


class Segment_Folds():
    def __init__(self, model_name, n_splits, model_path, class_num=4, tta_flag=False):
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
        self.tta_flag = tta_flag

        self.segment_models = list()
        self.get_segment_models()

    def get_segment_models(self):
        ''' 加载所有折的分割模型
        '''

        for fold in self.n_splits:
            self.segment_models.append(Get_Segment_Results(self.model_name, fold, self.model_path, self.class_num, tta_flag=self.tta_flag))

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