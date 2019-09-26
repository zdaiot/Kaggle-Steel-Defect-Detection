export http_proxy=http://localhost:8123
export https_proxy=http://localhost:8123

if [ $1 -eq 0 ]; then
    echo $1
    # 初始化元数据文件以创建数据集
    kaggle datasets init -p models
    # 更改默认的 json 文件，否则无法提交
    sed -i 's/INSERT_TITLE_HERE/models/g' models/dataset-metadata.json
    sed -i 's/INSERT_SLUG_HERE/models/g' models/dataset-metadata.json 
    # 创建一个新的数据集
    kaggle datasets create -p models
fi

if [ $1 -eq 1 ]; then
    # 更新数据集
    kaggle datasets version -p models -m "Updated data"
fi
