"""
本專案基於作者的檢測方法進行修改，參見
Shen et al., "Backdoor Scanning for Deep Neural Networks through K-Arm Optimization" (2021)
"""

import torch
import numpy as np
import random
import time
import os
from k_arm.utils import *
from k_arm.prescreening import pre_screening
from test import Pre_Screening_T, classes_matchingT
from k_arm.reverse import trigger_reverse_engineering
import pickle
import csv


SEED = 666
FILE_ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip\\id-00000102'
# FILE_ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip\\id-00000001'
MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')
PARAMS = {
    'init_cost': 1e-03,  # 或 1e-03
    'steps': 1000,
    'round': 60,
    'lr': 1e-01,
    'attack_success_threshold': 0.99,
    'patience': 5,
    'channels': 3,
    'batch_size': 32,
    'single_pixel_trigger_optimization': True,
    'epsilon': 1e-07,
    'bandits_epsilon': 0.3,
    'beta': 1e+4,
    'warmup_rounds': 2,
    'cost_multiplier': 1.5,
    'early_stop': False,
    'early_stop_threshold': 1,
    'early_stop_patience': 10,
    'central_init': True,
    'universal_attack_trigger_size_bound': 1720,
    'label_specific_attack_trigger_size_bound': 1000,
    'symmetric_check_bound': 10,
}

# if __name__ == '__main__':
    # print(f"{'-' * 40}掃描檔案: {FILE_ROOT_PATH}{'-' * 40}")
    #
    # StartTime = time.time()
    # setup_seed(SEED)
    # model = load_model(MODEL_PATH)
    #
    # print(f"{'*' * 20}Pre-Screening開始{'*' * 20}")
    #
    # # pre_screening會回傳過濾後可疑的target classes與victim classes
    # FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)
    #
    # # print(f"FilteredTargetClasses: {FilteredTargetClasses} {type(FilteredTargetClasses)}")
    # # print(f"FilteredVictimClasses: {FilteredVictimClasses} {type(FilteredVictimClasses)}")
    # print(f"{'*' * 20}Pre-Screening結束{'*' * 20}")
    #
    # target_classes, victim_classes, number_of_classes, backdoor_type = pre_process(
    #     FilteredTargetClasses, FilteredVictimClasses)
    # if backdoor_type is None:
    #     print('Model是安全的')
    # else:
    #     if backdoor_type == "universal":
    #         print(f'可能的攻擊方式: Universal Backdoor Attack')
    #         print(f'可能的 target class: {target_classes}')
    #         print(f'可能的 victim classes: ALL')
    #     else:
    #         print(f'可能的攻擊方式: Label Specific Backdoor Attack')
    #         candidates = []
    #         for i in range(number_of_classes):
    #             candidates.append(f'{target_classes[i]}-{victim_classes[i]}')
    #         print(f'可能的 target-victim 配對: {candidates}')
    #     print(f"{'*' * 20}Trigger Reverse Engineering開始{'*' * 20}")
    #     l1_norm, mask, target_class, victim_class, reverse_times = \
    #         trigger_optimization(target_classes, victim_classes, backdoor_type, model, DATA_PATH,
    #                              number_of_classes, 'forward', PARAMS)
    #     print(f"{'*' * 20}Trigger Reverse Engineering結束{'*' * 20}")
    #     print(f'Target Class: {target_class} Victim Class: {victim_class} '
    #           f'Trigger Size: {l1_norm} Optimization Steps: {reverse_times}')
    #
    #     PARAMS['step'] = reverse_times
    #     print(f"{'*' * 20}Symmetric Check開始{'*' * 20}")
    #     if backdoor_type == 'label specific':
    #         symmetric_l1_norm, _, _, _, _ = \
    #             trigger_optimization([victim_class.item()], torch.IntTensor([target_class]), backdoor_type,
    #                                  model, DATA_PATH, 1, 'backward', PARAMS)
    #     print(f"{'*' * 20}Symmetric Check結束{'*' * 20}")
    #
    #     TimeCost = time.time() - StartTime
    #     print(f"{'*' * 20}檢測結束{'*' * 20}")
    #     print(f"檢測結果:")
    #     if ((backdoor_type == 'universal' and PARAMS['universal_attack_trigger_size_bound'] > l1_norm)
    #             or (backdoor_type == 'label specific' and PARAMS['label_specific_attack_trigger_size_bound'] > l1_norm
    #                 and symmetric_l1_norm / l1_norm > PARAMS['symmetric_check_bound'])):
    #         print("Model含有後門")
    #     else:
    #         print("Model是安全的")
    #
    #     print(f"整體耗時: {TimeCost}")
    # 臨時
    # ---
    # temp_dict = {
    #     'target_classes': target_classes,
    #     'victim_classes': victim_classes,
    #     'backdoor_type': backdoor_type,
    #     'model': model,
    #     'DATA_PATH': DATA_PATH,
    #     'number_of_classes': number_of_classes,
    #     'forward': 'forward',
    #     'PARAMS': PARAMS,
    # }
    # with open('temp_var', 'wb') as f:
    #     pickle.dump(temp_dict, f)
    # ---
    #trigger_optimization(target_classes, victim_classes, backdoor_type, model, DATA_PATH, number_of_classes, 'forward', PARAMS)




if __name__ == '__main__':
    result = []
    count = 0
    for file in os.listdir('D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip'):
        count += 1
        if count == 51:
            break
        FILE_ROOT_PATH = os.path.join('D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip', file)
        MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
        DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')
        print(f"{'-' * 40}掃描檔案: {FILE_ROOT_PATH}{'-' * 40}")

        StartTime = time.time()
        setup_seed(SEED)
        model = load_model(MODEL_PATH)

        print(f"{'*' * 20}Pre-Screening開始{'*' * 20}")

        # pre_screening會回傳過濾後可疑的target classes與victim classes
        FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)

        print(f"{'*' * 20}Pre-Screening結束{'*' * 20}")

        target_classes, victim_classes, number_of_classes, backdoor_type = pre_process(
            FilteredTargetClasses, FilteredVictimClasses)
        if backdoor_type is None:
            print(f"{'*' * 20}檢測結束{'*' * 20}")
            print('檢測結果: Model是安全的(Benign)')
            result.append(0)
        else:
            if backdoor_type == "universal":
                print(f'可能的攻擊方式: Universal Backdoor Attack')
                print(f'可能的 target class: {target_classes.item()}')
                print(f'可能的 victim classes: ALL')
            else:
                print(f'可能的攻擊方式: Label Specific Backdoor Attack')
                candidates = []
                for i in range(number_of_classes):
                    candidates.append(f'{target_classes[i]}-{victim_classes[i]}')
                print(f'可能的 target-victim 配對: {candidates}')
            print(f"{'*' * 20}Trigger Reverse Engineering開始{'*' * 20}")
            l1_norm, mask, target_class, victim_class, reverse_times = \
                trigger_reverse_engineering(target_classes, victim_classes, backdoor_type, model, DATA_PATH,
                                            number_of_classes, 'forward', PARAMS)
            print(f"{'*' * 20}Trigger Reverse Engineering結束{'*' * 20}")
            print(f'Target Class: {target_class} Victim Class: {victim_class} '
                  f'Trigger Size: {l1_norm} Optimization Steps: {reverse_times}')

            PARAMS['step'] = reverse_times
            print(f"{'*' * 20}Symmetric Check開始{'*' * 20}")
            if backdoor_type == 'label specific':
                symmetric_l1_norm, _, _, _, _ = \
                    trigger_reverse_engineering([victim_class.item()], torch.IntTensor([target_class]), backdoor_type,
                                                model, DATA_PATH, 1, 'backward', PARAMS)
            print(f"{'*' * 20}Symmetric Check結束{'*' * 20}")

            TimeCost = time.time() - StartTime
            print(f"{'*' * 20}檢測結束{'*' * 20}")
            if ((backdoor_type == 'universal' and PARAMS['universal_attack_trigger_size_bound'] > l1_norm)
                    or (backdoor_type == 'label specific' and PARAMS[
                        'label_specific_attack_trigger_size_bound'] > l1_norm
                        and symmetric_l1_norm / l1_norm > PARAMS['symmetric_check_bound'])):
                print("檢測結果: Model含有後門(Abnormal)")
                result.append(1)
            else:
                print("檢測結果: Model是安全的(Benign)")
                result.append(0)

            print(f"整體耗時: {TimeCost}")
    with open('result.csv', 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=';')
        writer.writerows(result)
    print(result)
