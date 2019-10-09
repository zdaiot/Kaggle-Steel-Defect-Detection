import json, codecs
import os


def cal_thr_mean(file_name='result.json', model_name='unet_resnet34'):
    '''
    为了加快测试速度，有的时候使用多台电脑选阈值，这个时候需要手动计算阈值均值和像素阈值均值
    '''
    json_path = os.path.join('checkpoints', model_name, file_name)
    with open(json_path, 'r', encoding='utf-8') as json_file:
        result = json.load(json_file)

    n_splits = [0, 1, 2, 3, 4]
    best_thresholds_sum, best_minareas_sum, max_dices_sum = [0 for x in range(len(n_splits))], \
                                                            [0 for x in range(len(n_splits))], [0 for x in range(len(n_splits))]
    for fold in n_splits:
        best_thresholds, best_minareas, max_dices = result[str(fold)]['best_thresholds'], result[str(fold)]['best_minareas'], result[str(fold)]['max_dices']
        best_thresholds_sum = [x+y for x,y in zip(best_thresholds_sum, best_thresholds)]
        best_minareas_sum = [x+y for x,y in zip(best_minareas_sum, best_minareas)]
        max_dices_sum = [x+y for x,y in zip(max_dices_sum, max_dices)]
        
    best_thresholds_average, best_minareas_average, max_dices_average = [x / len(n_splits) for x in
                                                                         best_thresholds_sum], \
                                                                        [x / len(n_splits) for x in
                                                                         best_minareas_sum], [x / len(n_splits)
                                                                                              for x in
                                                                                              max_dices_sum]
    result['mean'] = {'best_thresholds': best_thresholds_average, 'best_minareas': best_minareas_average,
                       'max_dices': max_dices_average}
    print(result)
    with codecs.open(json_path, 'w', "utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False)


if __name__ == "__main__":
    cal_thr_mean()