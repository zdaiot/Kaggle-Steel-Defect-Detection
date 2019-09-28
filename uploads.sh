export http_proxy=http://localhost:8123
export https_proxy=http://localhost:8123

model_name="unet_resnet34"

# 建立文件夹，并copy py文件
if [ ! -d "kaggle/sources" ]; then
  mkdir -p kaggle/sources
fi
if [ ! -d "kaggle/sources/models" ]; then
  mkdir -p kaggle/sources/models
fi
if [ ! -d "kaggle/sources/datasets" ]; then
  mkdir -p kaggle/sources/datasets
fi

if [ ! -d "kaggle/checkpoints" ]; then
  mkdir -p kaggle/checkpoints
fi

rm kaggle/sources/*.py -f
rm kaggle/sources/*/*.py -f
rm checkpoints/$model_name/*_best.pth -f
rm checkpoints/$model_name/result.json -f

cp models/model.py kaggle/sources/models
cp datasets/steel_dataset.py kaggle/sources/datasets
cp solver.py kaggle/sources
cp classify_segment.py kaggle/sources

cp checkpoints/$model_name/*_best.pth kaggle/checkpoints
cp checkpoints/$model_name/result.json kaggle/checkpoints

if [ $1 -eq 0 ]; then
    echo "init uploads"
    # 初始化元数据文件以创建数据集
    kaggle datasets init -p kaggle/sources
    # 更改默认的 json 文件，否则无法提交
    sed -i 's/INSERT_TITLE_HERE/sources/g' kaggle/sources/dataset-metadata.json
    sed -i 's/INSERT_SLUG_HERE/sources/g' kaggle/sources/dataset-metadata.json
    # 创建一个新的数据集
    kaggle datasets create -p kaggle/sources -r zip

    # 初始化元数据文件以创建数据集
    kaggle datasets init -p kaggle/checkpoints
    # 更改默认的 json 文件，否则无法提交
    sed -i 's/INSERT_TITLE_HERE/checkpoints/g' kaggle/checkpoints/dataset-metadata.json
    sed -i 's/INSERT_SLUG_HERE/checkpoints/g' kaggle/checkpoints/dataset-metadata.json
    # 创建一个新的数据集
    kaggle datasets create -p kaggle/checkpoints
fi

if [ $1 -eq 1 ]; then
    echo "只更新脚本文件"
    # 更新脚本文件
    kaggle datasets version -p kaggle/sources -m "Updated Python Files"  -r zip
fi

if [ $1 -eq 2 ]; then
    echo "更新脚本文件和权重文件"
    # 更新数据集和脚本文件
    kaggle datasets version -p kaggle/sources -m "Updated Python Files"  -r zip
    kaggle datasets version -p kaggle/checkpoints -m "Updated Checkpoints"
fi
