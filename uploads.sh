#!/bin/bash

# 传参为0时，初始化；传参为1时，只更新脚本文件和kernel文件；传参为2时，更新脚本文件、kernel文件和权重文件
# 注意下面的model_name变量要根据情况修改
export http_proxy=http://localhost:8123
export https_proxy=http://localhost:8123

model_names="unet_resnet34 unet_resnet50 unet_se_resnext50_32x4d"

# 建立文件夹，用于存放上传到kaggle的文件
if [ ! -d "kaggle" ]; then
  mkdir -p kaggle/sources
  mkdir -p kaggle/sources/models
  mkdir -p kaggle/checkpoints
  mkdir -p kaggle/submission
  mkdir -p kaggle/segmentation_models
  # 复制依赖库
  cp models/*.tar.gz kaggle/segmentation_models
fi

# 删除已有的文件
rm kaggle/sources/*.py -f
rm kaggle/sources/*/*.py -f
rm kaggle/checkpoints/*_best.pth -f
rm kaggle/checkpoints/*result.json -f

# 复制脚本文件
cp models/model.py kaggle/sources/models
cp solver.py kaggle/sources
cp classify_segment.py kaggle/sources

# 创建__init__.py
echo "Create __init__.py"
touch kaggle/sources/models/__init__.py

# 复制权重文件
for model_name in $model_names; do 
    cp checkpoints/$model_name/*_best.pth kaggle/checkpoints
    cp checkpoints/$model_name/*result.json kaggle/checkpoints
done
cp checkpoints/result.json kaggle/checkpoints

# 复制kernel脚本
cp create_submission.py kaggle/submission/kernel.py

if [ $1 -eq 0 ]; then
    echo "init uploads"

    # 初始化依赖库
    kaggle datasets init -p kaggle/segmentation_models
    # 更改默认的 json 文件，否则无法提交
    sed -i 's/INSERT_TITLE_HERE/segmentation_models/g' kaggle/segmentation_models/dataset-metadata.json
    sed -i 's/INSERT_SLUG_HERE/segmentation_models/g' kaggle/segmentation_models/dataset-metadata.json
    # 创建一个新的数据集
    kaggle datasets create -p kaggle/segmentation_models

    # 初始化脚本文件
    kaggle datasets init -p kaggle/sources
    # 更改默认的 json 文件，否则无法提交
    sed -i 's/INSERT_TITLE_HERE/sources/g' kaggle/sources/dataset-metadata.json
    sed -i 's/INSERT_SLUG_HERE/sources/g' kaggle/sources/dataset-metadata.json
    # 创建一个新的数据集
    kaggle datasets create -p kaggle/sources -r zip

    # 初始化权重文件
    kaggle datasets init -p kaggle/checkpoints
    # 更改默认的 json 文件，否则无法提交
    sed -i 's/INSERT_TITLE_HERE/checkpoints/g' kaggle/checkpoints/dataset-metadata.json
    sed -i 's/INSERT_SLUG_HERE/checkpoints/g' kaggle/checkpoints/dataset-metadata.json
    # 创建一个新的数据集
    kaggle datasets create -p kaggle/checkpoints

    # 初始化kernel文件
    kaggle kernels init -p kaggle/submission
    sed -i 's/INSERT_KERNEL_SLUG_HERE/severstal-submission/g' kaggle/submission/kernel-metadata.json
    sed -i 's/INSERT_TITLE_HERE/severstal-submission/g' kaggle/submission/kernel-metadata.json
    sed -i 's/INSERT_CODE_FILE_PATH_HERE/kernel.py/g' kaggle/submission/kernel-metadata.json
    sed -i 's/Pick one of: {python,r,rmarkdown}/python/g' kaggle/submission/kernel-metadata.json
    sed -i 's/Pick one of: {script,notebook}/script/g' kaggle/submission/kernel-metadata.json
    USERNAME="$(kaggle config view | grep username:* | cut -c 13-)"
    sed -i 's/"enable_gpu": "false"/"enable_gpu": "true"/g' kaggle/submission/kernel-metadata.json

    datasets1="${USERNAME}/segmentation_models"
    datasets2="${USERNAME}/sources"
    datasets3="${USERNAME}/checkpoints"
    datasetsall="\"dataset_sources\": [\"${datasets1}\",\"${datasets2}\",\"${datasets3}\"]"
	# shellcheck disable=SC2154
	sed -i "s|\"dataset_sources\": \[\]|${datasetsall}|g" kaggle/submission/kernel-metadata.json
	sed -i 's#"competition_sources": \[\]#"competition_sources": \["severstal-steel-defect-detection"]#' kaggle/submission/kernel-metadata.json
    kaggle kernels push -p kaggle/submission
fi

if [ $1 -eq 1 ]; then
    echo "只更新脚本文件和kernel文件"
    # 更新脚本文件
    kaggle datasets version -p kaggle/sources -m "Updated Python Files"  -r zip
    kaggle kernels push -p kaggle/submission
fi

if [ $1 -eq 2 ]; then
    echo "更新脚本文件、kernel文件和权重文件"
    # 更新数据集和脚本文件
    kaggle datasets version -p kaggle/sources -m "Updated Python Files"  -r zip
    kaggle datasets version -p kaggle/checkpoints -m "Updated Checkpoints"
    kaggle kernels push -p kaggle/submission
fi
