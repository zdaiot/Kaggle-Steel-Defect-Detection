import json
import argparse
from argparse import Namespace


def get_seg_config():
    use_paras = False
    if use_paras:
        with open('./checkpoints/unet_resnet34/' + "params.json", 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)
        # dict to namespace
        config = Namespace(**config)
    else:
        parser = argparse.ArgumentParser()
        '''
        unet_resnet34时各个电脑可以设置的最大batch size
        zdaiot:12 z840:16 mxq:24
        unet_se_renext50
        hwp: 6 MXQ: 12
        unet_resnet50
        MXQ: 16
        '''
        # parser.add_argument('--image_size', type=int, default=768, help='image size')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        parser.add_argument('--epoch', type=int, default=50, help='epoch')

        parser.add_argument('--augmentation_flag', type=bool, default=True, help='if true, use augmentation method in train set')
        parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')
        parser.add_argument('--mask_only_flag', type=bool, default=True, help='if true, use masked data only.')
        parser.add_argument('--crop', type=bool, default=False, help='if true, crop image to [height, width].')
        parser.add_argument('--height', type=int, default=None, help='the height of cropped image')
        parser.add_argument('--width', type=int, default=None, help='the width of cropped image')

        # model set 
        parser.add_argument('--model_name', type=str, default='unet_resnet50', \
            help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4/unet_resnet50/unet_efficientnet_b3')

        # model hyper-parameters
        parser.add_argument('--class_num', type=int, default=4)
        parser.add_argument('--resume', type=str, default=0, help='Resuming from specified weight')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=5e-5, help='init lr')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
        
        # dataset 
        parser.add_argument('--save_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./datasets/Steel_data')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}

    return config


def get_classify_config():
    use_paras = False
    if use_paras:
        with open('./checkpoints/unet_resnet34/' + "params.json", 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)
        # dict to namespace
        config = Namespace(**config)
    else:
        parser = argparse.ArgumentParser()
        '''
        unet_resnet34时各个电脑可以设置的最大batch size
        zdaiot:12 z840:16 mxq:48
        unet_se_renext50
        hwp: 8
        unet_resnet50:
        MXQ: 24
        '''
        # parser.add_argument('--image_size', type=int, default=768, help='image size')
        parser.add_argument('--batch_size', type=int, default=24, help='batch size')
        parser.add_argument('--epoch', type=int, default=30, help='epoch')

        parser.add_argument('--augmentation_flag', type=bool, default=True, help='if true, use augmentation method in train set')
        parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')
        parser.add_argument('--crop', type=bool, default=False, help='if true, crop image to [height, width].')
        parser.add_argument('--height', type=int, default=None, help='the height of cropped image')
        parser.add_argument('--width', type=int, default=None, help='the width of cropped image')

        # model set 
        parser.add_argument('--model_name', type=str, default='unet_resnet50', \
            help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4/unet_resnet50/unet_efficientnet_b4')

        # model hyper-parameters
        parser.add_argument('--class_num', type=int, default=4)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=5e-5, help='init lr')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
        
        # dataset 
        parser.add_argument('--save_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./datasets/Steel_data')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}

    return config


if __name__ == '__main__':
    config = get_seg_config()
