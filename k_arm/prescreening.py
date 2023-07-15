import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from k_arm.dataset import CleanDataSet
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.25
UNIVERSAL_THETA = 0.65
LABEL_SPECIFIC_THETA = 0.9


def universal_backdoor_pre_scan(top_k_labels, num_of_classes):
    """
    針對 universal backdoor attack 的情境去 pre-scan ，檢測是否有符合該情境的 target label
    :param num_of_classes: model中classes的數量
    :param top_k_labels: 在結果中，有前 k 高 logit value 的 label
    :return TargetLabel: 輸出被判定為 target label 的 label；若沒有結果則會輸出-1
    """
    target_label = -1
    # labels_count: 每個 label 在所有 clean data 中出現的次數
    labels_count = np.array([top_k_labels[top_k_labels == i].shape[0] for i in range(num_of_classes)])
    max_count = np.max(labels_count)
    max_label = np.argmax(labels_count)
    if max_count > UNIVERSAL_THETA * top_k_labels.shape[0]:
        target_label = max_label
    return target_label


def label_specific_backdoor_pre_scan(top_k_labels, top_k_values, num_of_classes):
    sum_mat = torch.zeros(num_of_classes, num_of_classes)
    median_mat = torch.zeros(num_of_classes, num_of_classes)
    for i in range(num_of_classes):
        # class i
        top_k_i = top_k_labels[top_k_labels[:, 0] == i]
        top_k_i_pr = top_k_values[top_k_labels[:, 0] == i]
        top_k_j = torch.zeros(num_of_classes)
        for j in range(num_of_classes):
            # class j
            if i == j:
                top_k_j[j] = -1
            else:
                # 儲存label 為 i 的資料輸入模型時，結果中 j 的 logit 為top k的機率
                top_k_j[j] = top_k_i[top_k_i == j].shape[0] / top_k_i.shape[0]
                if top_k_j[j] >= LABEL_SPECIFIC_THETA:
                    sum_var = top_k_i_pr[top_k_i == j].sum()
                    median_var = torch.median(top_k_i_pr[top_k_i == j])
                    sum_mat[j, i] = sum_var
                    median_mat[j, i] = median_var
    return sum_mat, median_mat


def pre_screening(model, data_path):
    global AllLogits, AllPrs
    clean_data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CleanDataSet(file_path=data_path, transform=clean_data_transforms)

    CleanDataLoader = DataLoader(
        # 暫時先將num_workers設為0，解決方式: https://blog.csdn.net/JustPeanut/article/details/119146148
        # 主程式使用 if __name__ == 'main':
        dataset=dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
    )

    # 蒐集每個input輸入model後的 logits 與 prs
    for index, (image, label) in enumerate(CleanDataLoader):
        image, label = image.to(device), label.to(device)
        logits = model(image)
        prs = F.softmax(logits, 1)
        # 將所有的logits與pr都detach(不再計算梯度)與切換為CPU模式，使資源不被消耗殆盡
        if index == 0:
            AllLogits = logits.detach().cpu()
            AllPrs = prs.detach().cpu()
        else:
            AllLogits = torch.cat((AllLogits, logits.detach().cpu()), 0)
            AllPrs = torch.cat((AllPrs, prs.detach().cpu()), 0)

    # class 的數量
    num_of_classes = AllPrs.shape[1]

    # 檢查 universal attack 的攻擊情境
    # :在 theta% 的 clean data 中，假如 target label 的 logit value 皆在一個輸出結果中的前 gamma%
    # :就可以得到 universal attack 情境下的候選 target label

    # 在分類數量小於等於8時，k皆為2，否則0.25*classes個數可能為1，則相當於沒進行pre-screening
    if num_of_classes >= 8:
        k = math.floor(num_of_classes * GAMMA)
    else:
        k = 2
    # 選取出模型output的所有類別的 logits 中，最高的 k 個 logits(我們暫時用機率取代logits)
    # 使用 top_k
    top_k = torch.topk(AllPrs, k, dim=1)
    top_k_values = top_k[0]
    top_k_labels = top_k[1]

    target_label = universal_backdoor_pre_scan(top_k_labels, num_of_classes)
    if target_label == -1:
        # 檢查 label-specific attack 的情境
        target_matrix, median_matrix = label_specific_backdoor_pre_scan(top_k_labels, top_k_values, num_of_classes)
        target_classes = []
        triggered_classes = []
        for i in range(target_matrix.shape[0]):
            # 假如為可疑的組合:
            if target_matrix[i].sum() > 0:
                target_class = i
                triggered_class = target_matrix[i].nonzero().view(-1)
                triggered_class_pr = target_matrix[i][target_matrix[i] > 0]
                triggered_class_median = median_matrix[i][target_matrix[i] > 0]
                # 過濾掉pr與中位數沒過閥值的label，留下有通過的indexes
                # 機率總和與中位數需大於1e-8，FilteredIndexes為符合條件的indexes(該index為TriggeredClass的index)
                filtered_indexes = np.intersect1d((triggered_class_pr > 1e-8).nonzero().view(-1),
                                                  (triggered_class_median > 1e-8).nonzero().view(-1))
                if filtered_indexes.shape[0]:
                    triggered_class = triggered_class[filtered_indexes]
                    triggered_class_pr = triggered_class_pr[filtered_indexes]

                    # 由於此階段已排除 universal attack，所以設定一個閥值來避免太多不必要的 triggered label，此先設為3
                    if len(target_classes) > 3:
                        triggered_classes = triggered_classes[torch.topk(triggered_class_pr, 3, dim=0)]  # paper沒考慮中位數
                        # triggered_classes = triggered_classes[np.intersect1d(torch.topk(triggered_class_pr,3,dim=0)[1], torch.topk(triggered_class_median,3,dim=0)[1])] #也考慮中位數，之後跑跑看
                    target_classes.append(target_class)
                    triggered_classes.append(triggered_class)
        return target_classes, triggered_classes
    else:
        return target_label, None
