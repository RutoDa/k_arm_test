import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn as nn


class Scanner:
    def __init__(self, model, number_of_classes, params):
        self.cost_tensor = []
        self.cost = []  # 每個 arms 的 cost
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.init_cost = [params['init_cost']] * number_of_classes  # 每個 arms 的初始 cost
        self.steps = params['steps']
        self.round = params['round']
        self.lr = params['lr']
        self.number_of_classes = number_of_classes
        self.attack_success_threshold = params['attack_success_threshold']
        self.patience = params['patience']
        self.channels = params['channels']
        self.batch_size = params['batch_size']
        self.mask_size = (1, 224, 224)  # 可能要再否修改的可調整
        self.pattern_size = (1, 224, 224)  # 可能要再否修改
        self.single_pixel_trigger_optimization = params['single_pixel_trigger_optimization']


        # K-arms bandits(可修改點)
        self.epsilon = params['epsilon']
        self.bandits_epsilon = params['bandits_epsilon']
        self.beta = params['beta']
        self.warmup_rounds = params['warmup_rounds']
        self.cost_multiplier = params['cost_multiplier']
        self.early_stop_threshold = params['early_stop_threshold']
        self.early_stop_patience = params['early_stop_patience']

        self.mask_tanh_tensor = [torch.zeros(self.mask_size).to(self.device)] * self.number_of_classes  # k個(1,224,224)
        self.pattern_tanh_tensor = [torch.zeros(self.pattern_size).to(self.device)] * self.number_of_classes  # k個(1,224,224)
        self.pattern_raw_tensor = []
        self.mask_tensor = []
        # 找時間修改，有不必要性與重複性
        for i in range(self.number_of_classes):
            self.pattern_raw_tensor.append(torch.tanh(self.pattern_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)  # 都是0.5
            self.mask_tensor.append(torch.tanh(self.mask_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)

    def reset_state(self, init_pattern, init_mask):
        self.cost = self.init_cost
        self.cost_tensor = self.cost
        mask_np = init_mask.cpu().numpy()
        # 利用反函數去取 init_pattern 與 init_mask 經過處理前的樣子
        # 將mask裡的元素的範圍從[0,1]改至[-0.5,0.5]，並進行利用參數微調後取 tanh-1
        mask_tanh = np.arctanh((mask_np - 0.5) * (2 - self.epsilon))
        mask_tanh = torch.from_numpy(mask_tanh).to(self.device)
        # 將pattern裡的元素的範圍從[0,1]改至[-0.5,0.5]，並進行利用參數微調後取 tanh-1
        pattern_np = init_pattern.cpu().numpy()
        pattern_tanh = np.arctanh((pattern_np - 0.5) * (2 - self.epsilon))
        pattern_tanh = torch.from_numpy(pattern_tanh).to(self.device)

        for i in range(self.number_of_classes):
            self.mask_tanh_tensor[i] = mask_tanh.clone()
            self.mask_tanh_tensor[i].requires_grad = True
            self.pattern_tanh_tensor[i] = pattern_tanh.clone()
            self.pattern_tanh_tensor[i].requires_grad = True

    def update_tensor(self, mask_tanh_tensor, pattern_tanh_tensor, target_index, first_time=False):
        if first_time:
            for i in range(self.number_of_classes):
                self.mask_tensor[i] = torch.tanh(mask_tanh_tensor[i]) / (2 - self.epsilon) + 0.5
                self.pattern_raw_tensor[i] = torch.tanh(pattern_tanh_tensor[i]) / (2 - self.epsilon) + 0.5
        else:
            self.mask_tensor[target_index] = torch.tanh(mask_tanh_tensor[target_index]) / (2 - self.epsilon) + 0.5
            self.pattern_raw_tensor[target_index] = torch.tanh(pattern_tanh_tensor[target_index]) / (2 - self.epsilon) + 0.5

    def scanning(self, target_classes, data_loaders, target_index, init_pattern, init_mask, backdoor_type, direction):
        self.reset_state(init_pattern, init_mask)
        self.update_tensor(self.mask_tanh_tensor, self.pattern_tanh_tensor, target_index, first_time=True)

        best_mask = [None] * self.number_of_classes
        best_pattern = [None] * self.number_of_classes
        best_reg = [1e+10] * self.number_of_classes
        best_accuracy = [0] * self.number_of_classes

        log = []
        cost_set_counter = [0] * self.number_of_classes
        cost_down_counter = [0] * self.number_of_classes
        cost_down_flag = [False] * self.number_of_classes
        cost_up_counter = [0] * self.number_of_classes
        cost_up_flag = [False] * self.number_of_classes
        early_stop_counter = [0] * self.number_of_classes
        early_stop_reg_best = [1e+10] * self.number_of_classes
        early_stop_tag = [False] * self.number_of_classes
        update = [False] * self.number_of_classes

        avg_loss_ce = [1e+10] * self.number_of_classes  # average loss of cross entropy
        avg_loss_reg = [1e+10] * self.number_of_classes  # average loss of regularization
        avg_loss = [1e+10] * self.number_of_classes  # average loss
        avg_loss_acc = [1e+10] * self.number_of_classes  # average accuracy
        reg_down_vel = [-1e+10] * self.number_of_classes   # regularization 下降的速度
        times = [0] * self.number_of_classes
        total_times = [0] * self.number_of_classes
        first_best_reg = [1e+10] * self.number_of_classes
        target_tensor = torch.Tensor([target_classes[target_index]]).long().to(self.device)

        # optimizers 存放每個arms的 optimizer
        optimizers = []
        for i in range(self.number_of_classes):
            # 在 params 放入 trigger 的 pattern 與 mask
            optimizer = optim.Adam(
                [self.pattern_tanh_tensor[i], self.mask_tanh_tensor[i]], lr=self.lr, betas=(0.5, 0.9))
            optimizers.append(optimizer)

        pbar = tqdm(range(1000))  # !!!!!!!先用一次
        for step in pbar:
            target_tensor = torch.Tensor([target_classes[target_index]]).long().to(self.device)
            # print('target_tensor: ', target_tensor)
            total_times[target_index] += 1
            # print('total_times: ', total_times)
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for index, (images, labels) in enumerate(data_loaders[target_index]):
                images = images.to(self.device)
                target = target_tensor.repeat(images.shape[0])  # 使 target index 與 images 的 batch 數一致
                # print('target: ', target)
                # print('mask_tensor:', self.mask_tensor)
                # print('mask_tensor[i]:', self.mask_tensor[target_index].shape)
                triggered_input_tensor = (1-self.mask_tensor[target_index]) * images + self.mask_tensor[target_index] * self.pattern_raw_tensor[target_index]

                ####
                # if index == 0:
                #     plt.imshow(triggered_input_tensor[0].detach().cpu().permute(1, 2, 0))
                #     plt.show()
                # plt.imshow(images[0].detach().cpu().permute(1, 2, 0))
                # plt.show()
                # break
                optimizers[target_index].zero_grad()
                output_tensor = self.model(triggered_input_tensor)
                pred = output_tensor.argmax(dim=1, keepdim=True)
                loss_acc = pred.eq(target.view_as(pred)).sum().item() / images.shape[0]
                cross_entropy_loss = nn.CrossEntropyLoss()
                # print(output_tensor[0])
                loss_ce = cross_entropy_loss(output_tensor, target)
                loss_reg = torch.sum(torch.abs(self.mask_tensor[target_index]))

                # Trigger Optimizer 的 Loss
                loss = loss_ce + loss_reg * self.cost_tensor[target_index]
                loss.backward()
                optimizers[target_index].step()
                self.update_tensor(self.mask_tanh_tensor, self.pattern_tanh_tensor, target_index)

                pbar.set_description(
                    f'Target: {target_classes[target_index]}, victim: {labels[0]}, Loss: {loss:.4f},'
                    f' Acc: {loss_acc*100:.2f}%, CE_Loss: {loss_ce:.2f}, Reg_Loss:{loss_reg:.2f}, '
                    f'Cost:{self.cost_tensor[target_index]:.2f} best_reg:{best_reg[target_index]:.2f} '
                    f'avg_loss_reg:{avg_loss_reg[target_index]:.2f}')

                loss_ce_list.append(loss_ce.item())
                loss_reg_list.append(loss_reg.item())
                loss_list.append(loss.item())
                loss_acc_list.append(loss_acc)

