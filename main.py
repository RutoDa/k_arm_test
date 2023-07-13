# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import torch
import numpy as np
import random
import time

SEED = 666


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



if __name__ == '__main__':
    setup_seed(SEED)
