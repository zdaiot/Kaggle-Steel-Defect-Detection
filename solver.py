'''
该文件的功能：实现模型的前向传播，反向传播，损失函数计算，保存模型，加载模型功能
'''

import torch
import shutil
import os

class solver():
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
            masks_predict: [batch_size, class_num, height, width]，One-hot数据
        '''
        images = images.to(self.device)
        masks_predict = self.model(images)
        return masks_predict

    def cal_loss(self, masks, masks_predict, criterion):
        ''' 根据真实类标和预测出的类标计算损失
        
        Args:
            masks: [batch_size, class_num, height, width]，真实类标，One-hot数据
            masks_predict: [batch_size, class_num, height, width]，预测出的数据
            criterion: 使用的损失函数

        Return:
            loss: 计算出的损失值
        '''
        masks = masks.to(self.device)
        return criterion(masks_predict, masks)

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