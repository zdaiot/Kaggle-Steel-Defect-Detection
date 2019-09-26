import segmentation_models_pytorch as smp
import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
from utils.loss import ClassifyLoss


class ClassifyResNet(Module):
    def __init__(self, model_name, num_classes, training=True):
        super(ClassifyResNet, self).__init__()
        self.num_classes = num_classes
        if model_name == 'unet_resnet34':
            unet = smp.Unet('resnet34', encoder_weights='imagenet', classes=self.num_classes, activation=None)
        self.encoder = unet.encoder
        self.feature = nn.Conv2d(512, 32, kernel_size=1)
        self.logit = nn.Conv2d(32, self.num_classes, kernel_size=1)

        self.training = training

    def forward(self, x):
        x = self.encoder(x)[0]
        x = F.dropout(x, 0.5, training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)

        return logit



if __name__ == "__main__":
    class_net = ClassifyResNet('unet_resnet34', 4)
    x = torch.Tensor(8, 3, 256, 1600)
    y = torch.ones(8, 4)
    output = class_net(x)
    criterion = ClassifyLoss()
    loss = criterion(output, y)
    pass
