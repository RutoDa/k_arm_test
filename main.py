import torch
import numpy as np
import random
import time
import os
from k_arm.utils import *
from k_arm.prescreening import pre_screening

SEED = 666
FILE_ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round1\\TrainData\\models\\unzip\\id-00000102'
MODEL_PATH = os.path.join(FILE_ROOT_PATH, 'model.pt')
DATA_PATH = os.path.join(FILE_ROOT_PATH, 'clean-example-data')

if __name__ == '__main__':
    StartTime = time.time()
    setup_seed(SEED)
    model = load_model(MODEL_PATH)
    print(f"{'-'*20}Pre-Screening開始{'-'*20}")
    # pre_screening會回傳過濾後可疑的target classes與victim classes
    FilteredTargetClasses, FilteredVictimClasses = pre_screening(model, DATA_PATH)
    print(FilteredTargetClasses, FilteredVictimClasses)


    TimeCost = time.time() - StartTime
    print(f"{TimeCost}")