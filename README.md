Code for Kaggle [Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

## Datasets Prepare
Download steel datasets from [here](https://www.kaggle.com/c/severstal-steel-defect-detection/data) , unzip and put them into `../Input` directory.  

Structure of the `../Input` folder can be like:

```
test_images
train_images
sample_submission.csv
train.csv
```

Create soft links of datasets in the following directories:

```bash
cd Kaggle-Steel-Defect-Detection/datasets/Steel_data
ln -s ../../../Input/test_images ./
ln -s ../../../Input/train_images ./
ln -s ../../../Input/train.csv ./
ln -s ../../../Input/sample_submission.csv ./
```

## Requirements
* Install image augumentation library [albumentations](https://github.com/albu/albumentations)
```
conda install -c conda-forge imgaug
conda install albumentations -c albumentations
```
* Install [TensorBoard for Pytorch](https://pytorch.org/docs/stable/tensorboard.html)
```
pip install tb-nightly
pip install future
```
* Install [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
```
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

If you encountered error like: `ValueError: Duplicate plugins for name projector` when you are evacuating `tensorboard --logdir=checkpoints/unet_resnet34`, please refer to: [this](https://github.com/pytorch/pytorch/issues/22676).

```
I downloaded a test script from https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/tools/diagnose_tensorboard.py
I run it and it told me that I have two tensorboards with a different version. Also, it told me how to fix it.
I followed its instructions and I can make my tensorboard work.

I think this error means that you have two tensorboards installed so the plugin will be duplicated. Another method would be helpful that is to reinstall the python environment using conda.
```

## TODO
- [ ] 