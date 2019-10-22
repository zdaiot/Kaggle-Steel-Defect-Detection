import json, codecs
import os


def cal_thr_mean_splits(model_name='unet_resnet34', file_name='result.json', num_classes=4):
    """ 手动计算不同折之间的阈值均值和像素阈值均值

    :param model_name: 模型名称；类型为str
    :param file_name: 存放不同折结果的json文件名；类型为str
    :param num_classes: 有多少类数据
    :return: None
    """
    json_path = os.path.join('checkpoints', model_name, file_name)
    with open(json_path, 'r', encoding='utf-8') as json_file:
        result = json.load(json_file)

    n_splits = [0, 1, 2, 3, 4]
    best_thresholds_sum, best_minareas_sum, max_dices_sum = [0 for x in range(num_classes)], \
                                                            [0 for x in range(num_classes)], [0 for x in range(num_classes)]
    for fold in n_splits:
        best_thresholds, best_minareas, max_dices = result[str(fold)]['best_thresholds'], result[str(fold)]['best_minareas'], result[str(fold)]['max_dices']
        best_thresholds_sum = [x+y for x, y in zip(best_thresholds_sum, best_thresholds)]
        best_minareas_sum = [x+y for x, y in zip(best_minareas_sum, best_minareas)]
        max_dices_sum = [x+y for x, y in zip(max_dices_sum, max_dices)]
        
    best_thresholds_average, best_minareas_average, max_dices_average = [x / len(n_splits) for x in best_thresholds_sum],\
                                                                        [x / len(n_splits) for x in best_minareas_sum], \
                                                                        [x / len(n_splits) for x in max_dices_sum]
    result['mean'] = {'best_thresholds': best_thresholds_average, 'best_minareas': best_minareas_average,
                       'max_dices': max_dices_average}
    print(result)
    with codecs.open(json_path, 'w', "utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False)


def cal_thr_mean_models(segment_splits, file_name='result.json', num_classes=4):
    """

    :param segment_splits: 不同模型名称以及对应的fold；类型为dict
    :param file_name: 存放结果的json文件名；类型为str
    :param num_classes: 有多少类数据
    :return:
    """
    results = dict()
    best_thresholds_sum, best_minareas_sum, max_dices_sum = [0 for x in range(num_classes)], \
                                                            [0 for x in range(num_classes)], \
                                                            [0 for x in range(num_classes)]
    for model_name, fold in segment_splits.items():
        json_path = os.path.join('checkpoints', model_name, model_name + '_' + file_name)
        with open(json_path, 'r', encoding='utf-8') as json_file:
            result = json.load(json_file)

        best_thresholds, best_minareas, max_dices = result[str(fold)]['best_thresholds'], result[str(fold)][
            'best_minareas'], result[str(fold)]['max_dices']

        best_thresholds_sum = [x+y for x, y in zip(best_thresholds_sum, best_thresholds)]
        best_minareas_sum = [x+y for x, y in zip(best_minareas_sum, best_minareas)]
        max_dices_sum = [x+y for x, y in zip(max_dices_sum, max_dices)]

    best_thresholds_average, best_minareas_average, max_dices_average = [x / len(segment_splits) for x in best_thresholds_sum],\
                                                                        [x / len(segment_splits) for x in best_minareas_sum], \
                                                                        [x / len(segment_splits) for x in max_dices_sum]
    results['mean'] = {'best_thresholds': best_thresholds_average, 'best_minareas': best_minareas_average,
                       'max_dices': max_dices_average}
    print(results)

    with codecs.open(os.path.join('checkpoints', file_name), 'w', "utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False)


if __name__ == "__main__":
    # cal_thr_mean_splits()
    segment_splits = {'unet_resnet34': 1, 'unet_resnet50': 1, 'unet_se_resnext50_32x4d': 1}
    cal_thr_mean_models(segment_splits)
