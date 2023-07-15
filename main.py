import torch
import numpy as np
import random
import time
import os
from k_arm.utils import *
from k_arm.prescreening import pre_screening
from test import Pre_Screening_T

SEED = 666
FILE_ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip\\id-00000102'
MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')


def f(filtered_target_classes, filtered_victim_classes):
    try:
        if filtered_victim_classes is None:
            if filtered_target_classes is None:
                # 模型安全的(無target label，也無victim label)
                # target_classes = FilteredTargetClasses
                # victim_classes = FilteredVictimClasses
                number_of_classes = 0
                backdoor_type = None
                return filtered_target_classes, filtered_victim_classes, number_of_classes, backdoor_type
            else:
                # universal backdoor(因為pre-screening的輸出假如只有target label)
                number_of_classes = 1
                backdoor_type = 'universal'
                return filtered_target_classes, filtered_victim_classes, number_of_classes, backdoor_type
        else:
            # label specific backdoor
            target_classes, victim_classes = classes_matching(filtered_target_classes, filtered_victim_classes)
            num_classes = len(victim_classes)
            trigger_type = 'polygon_specific'

            print(f'Trigger Type: {trigger_type}')
            Candidates = []
            for i in range(len(target_classes)):
                Candidates.append('{}-{}'.format(target_classes[i], victim_classes[i]))
            print(f'Target-Victim Pair Candidates: {Candidates}')
    except Exception as e:
        print(f"ERROR1: {e}")


def target_victim_combination(target_classes, victim_classes):
    temp_target_classes = []
    temp_victim_classes = []

    for i in range(len(victim_classes)):
        target_classes_i = target_classes[i]
        victim_classes_i = victim_classes[i]
        for j in range(len(victim_classes_i)):
            temp_target_classes.append(target_classes_i)
            temp_victim_classes.append(victim_classes_i[j])

    return temp_target_classes, temp_victim_classes


if __name__ == '__main__':
    StartTime = time.time()
    setup_seed(SEED)
    model = load_model(MODEL_PATH)
    print(f"{'-' * 20}Pre-Screening開始{'-' * 20}")
    # pre_screening會回傳過濾後可疑的target classes與victim classes
    FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)


    TimeCost = time.time() - StartTime
    print(f"{TimeCost}")

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

