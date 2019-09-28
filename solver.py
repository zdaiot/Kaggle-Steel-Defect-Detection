'''
该文件的功能：实现模型的前向传播，反向传播，损失函数计算，保存模型，加载模型功能
'''

import torch
import shutil
import os


class Solver():
    def __init__(self, model):
        ''' 完成solver类的初始化
        Args:
            model: 网络模型
        '''
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images):
        ''' 实现网络的前向传播功能
        
        Args:
            images: [batch_size, channel, height, width]
            
        Return:
            output: 网络的输出，具体维度和含义与self.model有关，对我们任务而言：
                若self.model为分割模型，则维度为[batch_size, class_num, height, width]，One-hot数据
                若self.model为分类模型，则维度为[batch_size, class_num]，One-hot数据
        '''
        images = images.to(self.device)
        outputs = self.model(images)
        return outputs

    def cal_loss(self, targets, predicts, criterion):
        ''' 根据真实类标和预测出的类标计算损失
        
        Args:
            targets: 真实类标，具体维度和self.model有关，对我们任务而言：
                若为分割模型，则维度为[batch_size, class_num, height, width]，真实类标，One-hot数据
                若为分类模型，则维度为[batch_size, class_num]，真实类标，One-hot数据
            predicts: 网络的预测输出，具体维度和self.model有关，对我们任务而言：
                若为分割模型，则维度为[batch_size, class_num, height, width]，预测出的数据
                若为分类模型，则维度为[batch_size, class_num, 1, 1]，预测出的数据
            criterion: 使用的损失函数

        Return:
            loss: 计算出的损失值
        '''
        targets = targets.to(self.device)
        return criterion(predicts, targets)

    def backword(self, optimizer, loss):
        ''' 实现网络的反向传播
        
        Args:
            optimizer: 模型使用的优化器
            loss: 模型计算出的loss值
        Return:
            None
        '''
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def save_checkpoint(self, save_path, state, is_best):
        ''' 保存模型参数

        Args:
            save_path: 要保存的权重路径
            state: 存有模型参数、最大dice等信息的字典
            is_best: 是否为最优模型
        Return:
            None
        '''
        torch.save(state, save_path)
        if is_best:
            print('Saving Best Model.')
            save_best_path = save_path.replace('.pth', '_best.pth')
            shutil.copyfile(save_path, save_best_path)
    
    def load_checkpoint(self, load_path):
        ''' 保存模型参数

        Args:
            load_path: 要加载的权重路径
        
        Return:
            加载过权重的模型
        '''
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            self.model.module.load_state_dict(checkpoint['state_dict'])
            print('Successfully Loaded from %s' % (load_path))
            return self.model
        else:
            raise FileNotFoundError("Can not find weight file in {}".format(load_path))
