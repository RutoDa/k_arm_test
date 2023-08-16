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
from test import Pre_Screening_T, classes_matchingT, K_Arm_Opt
from k_arm.reverse import trigger_reverse_engineering
import pickle
import csv
from k_arm.parameter import PARAMS as P


SEED = 333
# FILE_ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip\\id-00000102'
# # FILE_ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip\\id-00000001'
# MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
# DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')



# # 實驗蒐集使用
# import warnings
# if __name__ == '__main__':
#     warnings.filterwarnings('ignore') # 關閉warnings使console乾淨
#
#
#     header = [['id', 'result', 'type', 'time cost', 'pre-screening', 'reverse', 'trigger size', 'sym ratio']]
#     with open('result.csv', 'w', newline='') as file:
#         w = csv.writer(file)
#         w.writerows(header)
#     count = 0
#     for file in os.listdir('D:\\UULi\\Datasets\\TrojAi\\Round3\\TrainData\\models\\unzip'):
#         count += 1
#         # if count < 740:
#         #     continue
#         # if file!='id-00000753':
#         #     continue
#
#         PARAMS = P.copy()
#         result = {'id': file}
#         FILE_ROOT_PATH = os.path.join('D:\\UULi\\Datasets\\TrojAi\\Round3\\TrainData\\models\\unzip', file)
#         MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
#         DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean_example_data')
#         print(f"{'-' * 40}掃描檔案: {FILE_ROOT_PATH}{'-' * 40}")
#
#         StartTime = time.time()
#         setup_seed(SEED)
#         model = load_model(MODEL_PATH)
#
#         print(f"{'*' * 3}Pre-Screening開始{'*' * 3}")
#
#         # pre_screening會回傳過濾後可疑的target classes與victim classes
#         FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)
#
#         print(f"{'*' * 3}Pre-Screening結束{'*' * 3}")
#         result['pre-screening'] = f'Targets: {FilteredTargetClasses}, Victims: {FilteredVictimClasses}'
#
#         target_classes, victim_classes, number_of_classes, backdoor_type = pre_process(
#             FilteredTargetClasses, FilteredVictimClasses)
#         if backdoor_type is None:
#             print(f"{'*' * 3}檢測結束{'*' * 3}")
#             print('檢測結果: Model是安全的(Benign)')
#             TimeCost = time.time() - StartTime
#             result['result'] = 'False'
#         else:
#             if backdoor_type == "universal":
#                 print(f'可能的攻擊方式: Universal Backdoor Attack')
#                 print(f'可能的 target class: {target_classes.item()}')
#                 print(f'可能的 victim classes: ALL')
#                 result['type'] = 'Universal'
#             else:
#                 print(f'可能的攻擊方式: Label Specific Backdoor Attack')
#                 candidates = []
#                 for i in range(number_of_classes):
#                     candidates.append(f'{target_classes[i]}-{victim_classes[i]}')
#                 print(f'可能的 target-victim 配對: {candidates}')
#                 result['type'] = 'label specific'
#             print(f"{'*' * 3}Trigger Reverse Engineering開始{'*' * 3}")
#             l1_norm, mask, target_class, victim_class, reverse_times = \
#                 trigger_reverse_engineering(target_classes, victim_classes, backdoor_type, model, DATA_PATH,
#                                             number_of_classes, 'forward', PARAMS)
#             print(f"{'*' * 3}Trigger Reverse Engineering結束{'*' * 3}")
#             print(f'Target Class: {target_class} Victim Class: {victim_class} '
#                   f'Trigger Size: {l1_norm} Optimization Steps: {reverse_times}')
#             result['reverse'] = f'Target Class: {target_class} Victim Class: {victim_class} Trigger Size: {l1_norm} ' \
#                                 f'Optimization Steps: {reverse_times}'
#             result['trigger size'] = l1_norm
#             PARAMS['steps'] = reverse_times
#
#             if backdoor_type == 'label specific' and PARAMS[
#                         'label_specific_attack_trigger_size_bound'] > l1_norm:
#                 print(f"{'*' * 3}Symmetric Check開始{'*' * 3}")
#                 symmetric_l1_norm, _, _, _, _ = \
#                     trigger_reverse_engineering([victim_class.item()], torch.IntTensor([target_class]), backdoor_type,
#                                                 model, DATA_PATH, 1, 'backward', PARAMS)
#                 result['sym ratio'] = symmetric_l1_norm / l1_norm
#                 print(f"{'*' * 3}Symmetric Check結束{'*' * 3}")
#
#             TimeCost = time.time() - StartTime
#             print(f"{'*' * 20}檢測結束{'*' * 20}")
#
#             if ((backdoor_type == 'universal' and PARAMS['universal_attack_trigger_size_bound'] > l1_norm)
#                     or (backdoor_type == 'label specific' and PARAMS[
#                         'label_specific_attack_trigger_size_bound'] > l1_norm
#                         and symmetric_l1_norm / l1_norm > PARAMS['symmetric_check_bound'])):
#                 print("檢測結果: Model含有後門(Abnormal)")
#                 result['result'] = 'True'
#             else:
#                 print("檢測結果: Model是安全的(Benign)")
#                 result['result'] = 'False'
#
#         print(f"整體耗時: {TimeCost}")
#         result['time cost'] = TimeCost
#         results = [[result.get('id', ''), result.get('result', ''), result.get('type', ''), result.get('time cost', '')
#                     , result.get('pre-screening', ''), result.get('reverse', ''), result.get('trigger size')
#                     , result.get('sym ratio', '')]]
#         with open('result.csv', 'a', newline='') as result_file:
#             writer = csv.writer(result_file)
#             writer.writerows(results)


# 實驗蒐集使用
import warnings
if __name__ == '__main__':
    warnings.filterwarnings('ignore') # 關閉warnings使console乾淨


    header = [['id', 'result', 'type', 'time cost', 'pre-screening', 'reverse', 'trigger size', 'sym ratio']]
    with open('result.csv', 'w', newline='') as file:
        w = csv.writer(file)
        w.writerows(header)
    count = 0
    for file in os.listdir('D:\\UULi\\Datasets\\TrojAi\\Round3\\TrainData\\models\\unzip'):
        count += 1
        # if count < 740:
        #     continue
        if file != 'id-00000013':
            continue

        PARAMS = P.copy()
        result = {'id': file}
        FILE_ROOT_PATH = os.path.join('D:\\UULi\\Datasets\\TrojAi\\Round3\\TrainData\\models\\unzip', file)
        MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
        DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean_example_data')
        print(f"{'-' * 40}掃描檔案: {FILE_ROOT_PATH}{'-' * 40}")

        StartTime = time.time()
        setup_seed(SEED)
        model = load_model(MODEL_PATH)

        print(f"{'*' * 3}Pre-Screening開始{'*' * 3}")

        # pre_screening會回傳過濾後可疑的target classes與victim classes
        FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)

        print(f"{'*' * 3}Pre-Screening結束{'*' * 3}")
        result['pre-screening'] = f'Targets: {FilteredTargetClasses}, Victims: {FilteredVictimClasses}'

        target_classes, victim_classes, number_of_classes, backdoor_type = pre_process(
            FilteredTargetClasses, FilteredVictimClasses)
        if backdoor_type is None:
            print(f"{'*' * 3}檢測結束{'*' * 3}")
            print('檢測結果: Model是安全的(Benign)')
            TimeCost = time.time() - StartTime
            result['result'] = 'False'
        else:
            if backdoor_type == "universal":
                print(f'可能的攻擊方式: Universal Backdoor Attack')
                print(f'可能的 target class: {target_classes.item()}')
                print(f'可能的 victim classes: ALL')
                result['type'] = 'Universal'
            else:
                print(f'可能的攻擊方式: Label Specific Backdoor Attack')
                candidates = []
                for i in range(number_of_classes):
                    candidates.append(f'{target_classes[i]}-{victim_classes[i]}')
                print(f'可能的 target-victim 配對: {candidates}')
                result['type'] = 'label specific'
            print(f"{'*' * 3}Trigger Reverse Engineering開始{'*' * 3}")
            l1_norm, mask, target_class, victim_class, reverse_times = \
                trigger_reverse_engineering(target_classes, victim_classes, backdoor_type, model, DATA_PATH,
                                            number_of_classes, 'forward', PARAMS)
            print(f"{'*' * 3}Trigger Reverse Engineering結束{'*' * 3}")
            print(f'Target Class: {target_class} Victim Class: {victim_class} '
                  f'Trigger Size: {l1_norm} Optimization Steps: {reverse_times}')
            result['reverse'] = f'Target Class: {target_class} Victim Class: {victim_class} Trigger Size: {l1_norm} ' \
                                f'Optimization Steps: {reverse_times}'
            result['trigger size'] = l1_norm
            PARAMS['steps'] = reverse_times

            if backdoor_type == 'label specific' and PARAMS[
                        'label_specific_attack_trigger_size_bound'] > l1_norm:
                # print(f"{'*' * 3}Symmetric Check開始{'*' * 3}")
                # symmetric_l1_norm, _, _, _, _ = \
                #     trigger_reverse_engineering([victim_class.item()], torch.IntTensor([target_class]), backdoor_type,
                #                                 model, DATA_PATH, 1, 'backward', PARAMS)
                # result['sym ratio'] = symmetric_l1_norm / l1_norm
                # print(f"{'*' * 3}Symmetric Check結束{'*' * 3}")
                print("檢測結果: Model含有後門(Abnormal)")
                result['result'] = 'True'
                symmetric_l1_norm = ''
            elif backdoor_type == 'universal' and PARAMS['universal_attack_trigger_size_bound'] > l1_norm:
                print("檢測結果: Model含有後門(Abnormal)")
                result['result'] = 'True'
                symmetric_l1_norm = ''
            else:
                print("檢測結果: Model是安全的(Benign)")
                result['result'] = 'False'
        TimeCost = time.time() - StartTime
        print(f"{'*' * 20}檢測結束{'*' * 20}")
        print(f"整體耗時: {TimeCost}")
        result['time cost'] = TimeCost
        results = [[result.get('id', ''), result.get('result', ''), result.get('type', ''), result.get('time cost', '')
                    , result.get('pre-screening', ''), result.get('reverse', ''), result.get('trigger size')
                    , result.get('sym ratio', '')]]
        with open('result.csv', 'a', newline='') as result_file:
            writer = csv.writer(result_file)
            writer.writerows(results)
