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
from test import Pre_Screening_T
from k_arm.optimization import trigger_optimization
import pickle


SEED = 666
FILE_ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip\\id-00000102'
MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')
PARAMS = {
    'init_cost': 0,  # 或 1e-03
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
    'cost_multiplier': 2,
    'early_stop_threshold': 1,
    'early_stop_patience': 10,
    'central_init': True,
}

if __name__ == '__main__':
    print(f"{'-' * 40}掃描檔案: {FILE_ROOT_PATH}{'-' * 40}")

    StartTime = time.time()
    setup_seed(SEED)
    model = load_model(MODEL_PATH)

    print(f"{'*' * 20}Pre-Screening開始{'*' * 20}")

    # pre_screening會回傳過濾後可疑的target classes與victim classes
    FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)

    print(f"FilteredTargetClasses: {FilteredTargetClasses}")
    print(f"FilteredVictimClasses: {FilteredVictimClasses}")
    print(f"{'*' * 20}Pre-Screening結束{'*' * 20}")

    target_classes, victim_classes, number_of_classes, backdoor_type = pre_process(
        FilteredTargetClasses, FilteredVictimClasses)
    if backdoor_type is None:
        print('Model是安全的(Benign)')
    elif backdoor_type == "universal":
        print(f'可能的攻擊方式: Universal Backdoor Attack')
        print(f'可能的 target class: {target_classes}')
        print(f'可能的 victim classes: ALL')
    else:
        print(f'可能的攻擊方式: Label Specific Backdoor Attack')
        candidates = []
        for i in range(number_of_classes):
            candidates.append(f'{target_classes[i]}-{victim_classes[i]}')
        print(f'可能的 target-victim 配對: {candidates}')

    # 臨時
    # ---
    temp_dict = {
        'target_classes': target_classes,
        'victim_classes': victim_classes,
        'backdoor_type': backdoor_type,
        'model': model,
        'DATA_PATH': DATA_PATH,
        'number_of_classes': number_of_classes,
        'forward': 'forward',
        'PARAMS': PARAMS,
    }
    with open('temp_var', 'wb') as f:
        pickle.dump(temp_dict, f)
    # ---
    #trigger_optimization(target_classes, victim_classes, backdoor_type, model, DATA_PATH, number_of_classes, 'forward', PARAMS)



    TimeCost = time.time() - StartTime
    print(f"{'*' * 20}檢測結束{'*' * 20}")
    print(f"整體耗時: {TimeCost}")

# if __name__ == '__main__':
#     StartTime = time.time()
#     setup_seed(SEED)
#     model = load_model(MODEL_PATH)
#     print(f"{'-' * 20}Pre-Screening開始{'-' * 20}")
#     # pre_screening會回傳過濾後可疑的target classes與victim classes
#     FilteredTargetClasses, FilteredVictimClasses = Pre_Screening_T(model, DATA_PATH, num_classes=5)
#     print('me:', pre_screening(model, DATA_PATH))
#     print(FilteredTargetClasses, FilteredVictimClasses)
#     print(f(FilteredTargetClasses, FilteredVictimClasses))
#     # print(classes_matching(FilteredTargetClasses, FilteredVictimClasses))
#
#     TimeCost = time.time() - StartTime
#     print(f"{TimeCost}")


# if __name__ == '__main__':
#     for file in os.listdir('D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip'):
#         FILE_ROOT_PATH = os.path.join('D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip', file)
#         MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
#         DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')
#         StartTime = time.time()
#         setup_seed(SEED)
#         model = load_model(MODEL_PATH)
#         print(f"{'-' * 20}{file}{'-' * 20}")
#         # pre_screening會回傳過濾後可疑的target classes與victim classes
#         FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)
#         FilteredTargetClasses_T, FilteredVictimClasses_T = Pre_Screening_T(model, DATA_PATH, num_classes=5)
#
#
#         print(f'FilteredTargetClasses: {FilteredTargetClasses}, FilteredVictimClasses: {FilteredVictimClasses}')
#         print(f'FilteredTargetClasses: {FilteredTargetClasses_T}, FilteredVictimClasses: {FilteredVictimClasses_T}')
#
#         assert FilteredTargetClasses == FilteredTargetClasses_T
#         #print(FilteredVictimClasses == FilteredVictimClasses_T)
#         #assert int(torch.all(FilteredVictimClasses == FilteredVictimClasses_T))
#
#         if FilteredVictimClasses_T is None:
#             if FilteredTargetClasses_T is None:
#                 number_of_classes = 0
#                 print(None)
#             else:
#                 # universal backdoor(因為pre-screening的輸出假如只有target label)
#                 number_of_classes = 1
#                 print('universal')
#         else:
#             # label specific backdoor
#             print('label spec')
#             x = classes_matchingT(FilteredTargetClasses_T, FilteredVictimClasses_T)
#             print(f'paper: {x}')
#         if FilteredVictimClasses is None:
#             if FilteredTargetClasses is None:
#                 number_of_classes = 0
#                 print('安全')
#             else:
#                 # universal backdoor(因為pre-screening的輸出假如只有target label)
#                 number_of_classes = 1
#                 print('universal')
#         else:
#             # label specific backdoor
#             print('label spec')
#             y = classes_matchingT(FilteredTargetClasses, FilteredVictimClasses)
#             print(f'me: {y}')
#             assert x == y
#
#
#         TimeCost = time.time() - StartTime
#         print(f"{TimeCost}")

