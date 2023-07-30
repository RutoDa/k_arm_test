import torch
from k_arm.dataset import CleanDataSet, TrojAI_transform
from torch.utils.data import DataLoader
import torch.nn.functional as F
from k_arm.scanner import Scanner
import pickle
# import test


def trigger_reverse_engineering(
        target_classes, victim_classes, backdoor_type, model, data_path, number_of_classes, direction, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 針對不同的攻擊情境建立不同數量的ARMs
    data_loaders = []
    if victim_classes is None:
        # Universal backdoor attack
        clean_data_set = CleanDataSet(file_path=data_path, transform=TrojAI_transform, victim_classes=victim_classes)
        data_loader = DataLoader(
            dataset=clean_data_set, batch_size=params['batch_size'], num_workers=8, pin_memory=True)
        data_loaders.append(data_loader)
    else:
        # Label Specific backdoor attack
        for i in range(len(victim_classes)):
            clean_data_set = CleanDataSet(
                file_path=data_path, transform=TrojAI_transform, victim_classes=victim_classes[i], label_specific=True)
            data_loader = DataLoader(
                dataset=clean_data_set, batch_size=params['batch_size'], num_workers=0, pin_memory=True)
            data_loaders.append(data_loader)

    scanner = Scanner(model, number_of_classes, params)
    # scanner = test.K_Arm_Scanner(model, number_of_classes)

    # 初始化 pattern
    if params['single_pixel_trigger_optimization'] and backdoor_type == 'label specific':
        # 檢測 single pixel 的攻擊
        pattern = torch.rand(1, params['channels'], 1, 1).to(device)
    else:
        pattern = torch.rand(1, params['channels'], 224, 224).to(device)  # 寬跟高之後修改，使不會寫死
    #pattern = torch.clamp(pattern, min=0, max=1)  # 將 pattern 夾到 0~1 之間 (這句感覺不必要使用)

    # 初始化 mask
    if backdoor_type == 'universal':
        mask = torch.rand(1, 224, 224).to(device)  # 寬跟高之後修改，使不會寫死
    elif backdoor_type == 'label specific':
        if params['central_init']:
            mask = torch.rand(1, 224, 224).to(device) * 0.001
            mask[:, 112 - 25:112 + 25, 112 - 25:112 + 25] = 0.99  # 將正中間的區域塗上顏色
        else:
            mask = torch.rand(1, 224, 224).to(device)  # 寬跟高之後修改，使不會寫死
    #mask = torch.clamp(mask, min=0, max=1)  # 將 mask 夾到 0~1 之間 (這句感覺不必要使用)

    start_index = 0
    pattern, mask, l1_norm, time_cost = scanner.scanning(
        target_classes, data_loaders, start_index, pattern, mask, backdoor_type, direction)


    # pattern, mask, l1_norm, time_cost = scanner.scanning(
    #     target_classes, data_loaders, start_index, pattern, mask, backdoor_type, direction)


    index = torch.argmin(torch.Tensor(l1_norm))

    if victim_classes is None:
        target_class = target_classes[index]
        victim_class = 'all'
    else:
        target_class = target_classes[index]
        victim_class = victim_classes[index]
    return l1_norm[index], mask[index], target_class, victim_class, time_cost[index]


if __name__ == '__main__':
    with open('../temp_var', 'rb') as f:
        temp_var = pickle.load(f)
    trigger_reverse_engineering(temp_var['target_classes'], temp_var['victim_classes'], temp_var['backdoor_type'],
                                temp_var['model'], temp_var['DATA_PATH'], temp_var['number_of_classes'], 'forward',
                                temp_var['PARAMS'])
