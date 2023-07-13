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
