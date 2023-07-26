import torch
import numpy as np
import random


def setup_seed(seed):
    """
    設置所有module的cpu或gpu的seed，使之後可以再次實現實驗結果
    :param seed: 隨機種子
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model_path):
    """
    載入指定路徑中的model並回傳
    :param model_path: model.pt的路徑
    :return: model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model.to(device)
    model.eval()  # 切換到評估模式
    return model


def pre_process(filtered_target_classes, filtered_victim_classes):
    """
    在開始執行Trigger Optimization的pre-process，會輸出最終要檢查的target與victim的相對應組合，並輸出判定的後門類別
    :param filtered_target_classes: 所有通過pre-screening後，篩選的target class的集合list
    :param filtered_victim_classes: 所有通過pre-screening後，篩選的victim class的集合list
    :return: 經過對應組合後的 target_classes(若無後門則為None) 與 victim_classes(若是universal或無後門則為None)
    """
    if filtered_victim_classes is None:
        # universal backdoor(因為pre-screening的輸出假如只有target label)
        number_of_classes = 1
        backdoor_type = 'universal'
        return filtered_target_classes, filtered_victim_classes, number_of_classes, backdoor_type
    elif not filtered_target_classes:
        backdoor_type = None
        return None, None, None, backdoor_type
    else:
        # label specific backdoor
        # 將所有可能的 target label 與 victim label 配對組合出來(target_classes與victim_classes中的元素互相對應)
        target_classes, victim_classes = target_victim_combination(filtered_target_classes, filtered_victim_classes)
        number_of_classes = len(victim_classes)
        assert len(target_classes) == len(victim_classes)  # 測試用
        backdoor_type = 'label specific'
        return target_classes, victim_classes, number_of_classes, backdoor_type


def target_victim_combination(target_classes, victim_classes):
    """
    將所有可能的 target label 與 victim label 配對組合出來(target_classes與victim_classes中的元素互相對應)
    :param target_classes: 所有可能為target class的集合list
    :param victim_classes: 所有可能為victim class的集合list
    兩個param的元素需互相對應
    :return: 所有可能的配對組合(temp_target_classes, temp_victim_classes的元素乎相對應)
    """
    temp_target_classes = []
    temp_victim_classes = []

    for i in range(len(victim_classes)):
        target_classes_i = target_classes[i]
        victim_classes_i = victim_classes[i]
        for j in range(len(victim_classes_i)):
            temp_target_classes.append(target_classes_i)
            temp_victim_classes.append(victim_classes_i[j])

    return temp_target_classes, temp_victim_classes
