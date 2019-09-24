Code for Kaggle [Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

## Datasets Prepare
Download steel datasets from [here](https://www.kaggle.com/c/severstal-steel-defect-detection/data) , unzip and put them into `../input` directory.  

Structure of the `../input` folder can be like:

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
```

## Requirements

## TODO
- [ ] 