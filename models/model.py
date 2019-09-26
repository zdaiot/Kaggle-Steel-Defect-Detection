import segmentation_models_pytorch as smp
import torch

class Model():
    ''' 根据 model_name 初始化模型，并返回模型
    
    依赖：pip install git+https://github.com/qubvel/segmentation_models.pytorch
    '''
    def __init__(self, model_name, encoder_weights='imagenet', class_num=4):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_num = class_num
        self.encoder_weights = encoder_weights
    
    def create_model(self):
        print("Using model: {}".format(self.model_name))

        # Uent resnet系列
        if self.model_name == 'unet_resnet34':
            self.model = smp.Unet('resnet34', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        elif self.model_name == 'unet_resnet50':
            self.model = smp.Unet('resnet50', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        # Unet resnext系列
        elif self.model_name == 'unet_resnext50_32x4d':
            self.model = smp.Unet('resnext50_32x4d', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        # Unet se_resnet系列
        elif self.model_name == 'unet_se_resnet50':
            self.model = smp.Unet('se_resnet50', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        # Unet se_resnext 系列
        elif self.model_name == 'unet_se_resnext50_32x4d':
            self.model = smp.Unet('se_resnext50_32x4d', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        # Unet dpn 系列
        elif self.model_name == 'unet_dpn68':
            self.model = smp.Unet('dpn68', encoder_weights=self.encoder_weights, classes=self.class_num, activation=None)
        
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        return self.model


if __name__ == "__main__":
    model_name = 'unet_resnet34'
    model = Model(model_name).create_model()
    print(model)