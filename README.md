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

## online submission of local csv file
Please see [this](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/108638#latest-628715) for detailed information

First, Install [Kaggle API](https://github.com/Kaggle/kaggle-api): `pip install kaggle`

To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json

For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Clone the repo: `git clone https://github.com/alekseynp/kaggle-dev-ops.git`
Enter the repo: `cd kaggle-dev-ops`
Go to severstal: `cd severstal-steel-defect-detection`
Initialize: `make init-csv-submission`
Submit: `SUBMISSION=/path/to/csv/file.csv make release-csv`
Click the link to the kernel and press the submit to competition button.

When run `SUBMISSION=/path/to/csv/file.csv make release-csv`, If you encounter the following erro: `Invalid dataset specification /severstal_csv_submission`. You should manually edit the `kernel-csv-metadata.json` and add your username here:
"dataset_sources": ["YOUR_KAGGLE_USERNAME_HERE/severstal_csv_submission"],

**Please notice that:** Any submission made with this tool will score zero on the final private LB. The point of the tool is to make it easy to quickly submit CSVs created locally for the public test set and get a public LB score.

## TODO
- [x] finish classify + segment model
- [x] finish create_submission.py 
- [x] finish demo.py
- [x] finish loss.py
- [x] finish choose_threshold
- [x] finish data augmentation
- [ ] EfficientB4( w/ ASPP)
- [x] ResNet50
- [x] code review(validation dice, threshold dice)
- [ ] choose fold
- [ ] ensemble
- [x] early stopping automaticly
- [ ] GN