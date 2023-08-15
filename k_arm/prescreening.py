import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from k_arm.dataset import CleanDataSet
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.25
UNIVERSAL_THETA = 0.95  # 0.65
LABEL_SPECIFIC_THETA = 0.9


def universal_backdoor_pre_scan(top_k_labels, num_of_classes):
    """
    針對 universal backdoor attack 的情境去 pre-scan，檢測是否有符合該情境的 target label
    細節:
    假如 theta% 的 output 中，有某個 label 都在 array top_k_labels 中，
    則判定該模型可能有 universal 的攻擊，並回傳該可疑的 target label。
    :param num_of_classes: model 中 classes 的數量
    :param top_k_labels: 在結果中，有前 k 高 logit value 的 label
    :return TargetLabel: 輸出被判定為 target label 的 label；若沒有結果則會輸出-1
    """
    target_label = -1
    # labels_count: 每個 label 在所有 clean data 的top k中出現的次數
    labels_count = np.array([top_k_labels[top_k_labels == i].shape[0] for i in range(num_of_classes)])
    max_count = np.max(labels_count)
    max_label = np.argmax(labels_count)
    if max_count > UNIVERSAL_THETA * top_k_labels.shape[0]:
        target_label = max_label
    return torch.tensor(target_label).unsqueeze(0)


def label_specific_backdoor_pre_scan(top_k_labels, top_k_values, num_of_classes):
    """
    針對 label specific backdoor attack 的情境去 pre-scan，檢測是否有符合該情境的 target labels 與其相對應的 victim labels
    細節:
    假如 v 的照片輸入模型後有 theta% 的結果中，t 皆在 array topk 中，則判定該模型可能有label specific得攻擊，
    並回傳該可疑的target labels 與相對應的 victim labels
    :param top_k_labels: 在結果中，有前 k 高 logit value 的 label
    :param top_k_values: 在結果中，有前 k 高 logit value 的 label 的 機率
    :param num_of_classes: classes 的數量
    :return:
    """
    # sum_matrix 每個 row 代表不同的 target label，col分別為對應到的 victim label，
    # 而 sum_matrix[j][i]中儲存 top_k_labels 中符合門檻值的 [i,j] 的機率總和
    sum_matrix = torch.zeros(num_of_classes, num_of_classes)
    median_matrix = torch.zeros(num_of_classes, num_of_classes)
    # 將 top_k_labels 上的元素視為[[col 0, col 1],...]
    for i in range(num_of_classes):
        # class i
        # top_k_i 為 top_k_labels 中 col 0 為 i 的元素集合
        top_k_i = top_k_labels[top_k_labels[:, 0] == i]
        # top_k_i_pr 為 top_k_values 中 col 0 為 i 的元素集合
        top_k_i_pr = top_k_values[top_k_labels[:, 0] == i]

        for j in range(num_of_classes):
            # class j
            if not i == j:
                # top_k_i_j_pr 為 label 為 i 的資料輸入模型時，結果中 j 的 logit 為top k的機率
                top_k_i_j_pr = top_k_i[top_k_i == j].shape[0] / top_k_i.shape[0]
                if top_k_i_j_pr >= LABEL_SPECIFIC_THETA:
                    sum_var = top_k_i_pr[top_k_i == j].sum()
                    median_var = torch.median(top_k_i_pr[top_k_i == j])
                    sum_matrix[j, i] = sum_var  # 儲存在clean input 為 i 的結果中，j 在結果為第二高時的機率的總和
                    median_matrix[j, i] = median_var  # 儲存在clean input 為 i 的結果中，j 在結果為第二高時的機率的中位數

    target_classes = []
    triggered_classes = []
    for i in range(sum_matrix.shape[0]):
        # 假如為可疑的組合:
        if sum_matrix[i].sum() > 0:
            target_class = i
            triggered_class = sum_matrix[i].nonzero().view(-1)  # 在 target class 為 i 時的所有可能 victims 的 indexes
            triggered_class_pr = sum_matrix[i][sum_matrix[i] > 0]  # 與 triggered_class 中每個元素對應的機率
            triggered_class_median = median_matrix[i][sum_matrix[i] > 0]  # 與 triggered_class 中每個元素對應的中位數
            # 過濾掉pr與中位數沒過閥值的label，留下有通過的indexes
            # 機率總和與中位數需大於1e-8，FilteredIndexes為符合條件的indexes(該index為TriggeredClass的index)
            filtered_indexes = np.intersect1d((triggered_class_pr > 1e-8).nonzero().view(-1),
                                              (triggered_class_median > 1e-8).nonzero().view(-1))  # 由1e-8降低
            if filtered_indexes.shape[0]:
                triggered_class = triggered_class[filtered_indexes]
                triggered_class_pr = triggered_class_pr[filtered_indexes]

                # 由於此階段已排除 universal attack，所以設定一個閥值來避免太多不必要的 triggered label，此先設為3
                if len(triggered_class) > 3:
                    triggered_class = triggered_class[torch.topk(triggered_class_pr, 3, dim=0)[1]]  # paper沒考慮中位數
                    # triggered_classes = triggered_classes[np.intersect1d(torch.topk(triggered_class_pr,3,dim=0)[1], torch.topk(triggered_class_median,3,dim=0)[1])] #也考慮中位數，之後跑跑看
                target_classes.append(target_class)
                triggered_classes.append(triggered_class)
    return target_classes, triggered_classes


def pre_screening(model, data_path):
    """
    透過pre_screening可以經由以下步驟篩選出可能的target labels與victim labels
    1. 載入所有的clean sample(各個分類的正常input照片)
    2. 將所有 clean sample 都 input 到 model，並將output都蒐集好
    3. 將output都使用softmax將logit轉成機率
    4. 將找出每個output中top k(k根據參數gamma所定)的機率，存到 array topk
    5. 判斷哪些 label 可能遭受攻擊
        - Universal backdoor attack
            - 假如theta%的output中，有某個label都在array topk 中，則判定該模型可能有universal得攻擊，並回傳該可疑的target label
        - Label specific backdoor attack
            - 假設有一對 target label(t) 與 victim label(v)，
            - 假如 v 的照片輸入模型後有 theta% 的結果中，t 皆在 array topk 中，則判定該模型可能有label specific得攻擊，
                並回傳該可疑的target labels 與相對應的 victim labels
    :param model:
    :param data_path:
    :return:
    假如為 universal attack，
    則回傳 target_label(可能的target label), None
    假如為 label specific attack，
    則回傳 target_classes(可能為target class的index), triggered_classes(target_classes相對應的victim class的index)
    若皆不符合，則會回傳 [], []
    """
    global AllLogits, AllPrs
    clean_data_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = CleanDataSet(file_path=data_path, transform=clean_data_transforms)

    CleanDataLoader = DataLoader(
        # 暫時先將num_workers設為0，解決方式: https://blog.csdn.net/JustPeanut/article/details/119146148
        # 主程式使用 if __name__ == 'main':
        dataset=dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True  # num_workers=4
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
        # k = math.floor(num_of_classes * GAMMA)
        k = round(num_of_classes * GAMMA)
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
        target_classes, triggered_classes = label_specific_backdoor_pre_scan(top_k_labels, top_k_values, num_of_classes)
        return target_classes, triggered_classes
    else:
        return target_label, None
