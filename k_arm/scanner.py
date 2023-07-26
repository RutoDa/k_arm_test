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
        self.mask_size = [1, 224, 224]  # 可能要再否修改的可調整
        self.pattern_size = [1, 3, 224, 224]  # 可能要再否修改
        self.single_pixel_trigger_optimization = params['single_pixel_trigger_optimization']

        # K-arms bandits(可修改點)
        self.epsilon = params['epsilon']
        self.bandits_epsilon = params['bandits_epsilon']
        self.beta = params['beta']
        self.warmup_rounds = params['warmup_rounds']
        self.cost_multiplier = params['cost_multiplier']
        self.early_stop = params['early_stop']
        self.early_stop_threshold = params['early_stop_threshold']
        self.early_stop_patience = params['early_stop_patience']
        self.mask_tanh_tensor = [torch.zeros(self.mask_size).to(self.device)] * self.number_of_classes  # k個(1,224,224)
        self.pattern_tanh_tensor = [torch.zeros(self.pattern_size).to(
            self.device)] * self.number_of_classes  # k個(1,224,224)
        self.pattern_raw_tensor = []
        self.mask_tensor = []
        # 找時間修改，有不必要性與重複性
        for i in range(self.number_of_classes):
            self.pattern_raw_tensor.append(torch.tanh(self.pattern_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)  # 都是0.5
            self.mask_tensor.append(torch.tanh(self.mask_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)

    def reset_state(self, init_pattern, init_mask):
        self.cost = [0] * self.number_of_classes
        self.cost_tensor = self.cost
        # self.cost = self.init_cost
        # self.cost_tensor = self.cost
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
            self.pattern_raw_tensor[target_index] = torch.tanh(pattern_tanh_tensor[target_index]) / (
                    2 - self.epsilon) + 0.5

    def scanning(self, target_classes, data_loaders, target_index, init_pattern, init_mask, backdoor_type, direction):
        self.reset_state(init_pattern, init_mask)
        self.update_tensor(self.mask_tanh_tensor, self.pattern_tanh_tensor, target_index, first_time=True)

        best_mask = [None] * self.number_of_classes
        best_pattern = [None] * self.number_of_classes
        best_reg = [1e+10] * self.number_of_classes
        best_accuracy = [0] * self.number_of_classes

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
        reg_down_vel = [-1e+10] * self.number_of_classes  # regularization 下降的速度
        times = [0] * self.number_of_classes
        total_times = [0] * self.number_of_classes
        first_best_reg = [1e+10] * self.number_of_classes

        # optimizers 存放每個arms的 optimizer
        optimizers = []
        for i in range(self.number_of_classes):
            # 在 params 放入 trigger 的 pattern 與 mask
            optimizer = optim.Adam(
                [self.pattern_tanh_tensor[i], self.mask_tanh_tensor[i]], lr=self.lr, betas=(0.5, 0.9))
            optimizers.append(optimizer)
        cross_entropy_loss = nn.CrossEntropyLoss()
        pbar = tqdm(range(1000))  # !!!!!!!先用10次
        for step in pbar:
            target_tensor = torch.Tensor([target_classes[target_index]]).long().to(self.device)
            total_times[target_index] += 1
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for images, labels in data_loaders[target_index]:
                images = images.to(self.device)
                target = target_tensor.repeat(images.shape[0])  # 使 target index 與 images 的 batch 數一致
                triggered_input_tensor = (1 - self.mask_tensor[target_index]) * images + self.mask_tensor[
                    target_index] * self.pattern_raw_tensor[target_index]

                ####
                # if step % 10 == 0:
                #     plt.imshow(triggered_input_tensor[0].detach().cpu().permute(1, 2, 0))
                #     # plt.show()
                #     #plt.savefig(f'../trigger_img/{step}.png')
                ####
                optimizers[target_index].zero_grad()
                output_tensor = self.model(triggered_input_tensor)
                pred = output_tensor.argmax(dim=1, keepdim=True)
                loss_acc = pred.eq(target.view_as(pred)).sum().item() / images.shape[0]
                loss_ce = cross_entropy_loss(output_tensor, target)
                loss_reg = torch.sum(torch.abs(self.mask_tensor[target_index]))

                # Trigger Optimizer 的 Loss
                loss = loss_ce + loss_reg * self.cost_tensor[target_index]
                loss.backward()
                optimizers[target_index].step()
                self.update_tensor(self.mask_tanh_tensor, self.pattern_tanh_tensor, target_index)

                pbar.set_description(
                    f'Target: {target_classes[target_index]}, victim: {labels[0]}, Loss: {loss:.4f},'
                    f' Acc: {loss_acc * 100:.2f}%, CE_Loss: {loss_ce:.2f}, Reg_Loss:{loss_reg:.2f}, '
                    f'Cost:{self.cost_tensor[target_index]:.2f} best_reg:{best_reg[target_index]:.2f} '
                    f'avg_loss_reg:{avg_loss_reg[target_index]:.2f}')

                loss_ce_list.append(loss_ce.item())
                loss_reg_list.append(loss_reg.item())
                loss_list.append(loss.item())
                loss_acc_list.append(loss_acc)

            # K-ARM Bandits 演算
            avg_loss_ce[target_index] = np.mean(loss_ce_list)
            avg_loss_reg[target_index] = np.mean(loss_reg_list)
            avg_loss[target_index] = np.mean(loss_list)
            avg_loss_acc[target_index] = np.mean(loss_acc_list)

            if avg_loss_acc[target_index] > best_accuracy[target_index]:
                best_accuracy[target_index] = avg_loss_acc[target_index]

            if direction == 'forward':
                if backdoor_type == 'label specific' and (
                        (total_times[target_index] > 20 and best_accuracy[target_index] < 0.3)
                        or (total_times[target_index] > 200 and best_accuracy[target_index] < 0.8)
                        or (total_times[target_index] > 10 and best_accuracy[target_index] == 0)):
                    early_stop_tag[target_index] = True
            elif direction == 'backward':
                if (backdoor_type == 'label specific' and
                        total_times[target_index] > 200 and best_accuracy[target_index] < 1):
                    early_stop_tag[target_index] = True

            update[target_index] = False
            if avg_loss_acc[target_index] >= self.attack_success_threshold and avg_loss_reg[target_index] < best_reg[
                target_index]:
                best_mask[target_index] = self.mask_tensor[target_index]
                best_pattern[target_index] = self.pattern_raw_tensor[target_index]
                update[target_index] = True
                times[target_index] += 1

                if times[target_index] == 1:
                    first_best_reg[target_index] = 2500
                reg_down_vel[target_index] = (first_best_reg[target_index] - avg_loss_reg[target_index]) / (
                        times[target_index] + (total_times[target_index] / 2))
                best_reg[target_index] = avg_loss_reg[target_index]

            if self.early_stop:
                if best_reg[target_index] < 1e+10:
                    if best_reg[target_index] >= self.early_stop_threshold * early_stop_reg_best[target_index]:
                        early_stop_counter[target_index] += 1
                    else:
                        early_stop_counter[target_index] = 0
                early_stop_reg_best[target_index] = min(best_reg[target_index], early_stop_reg_best[target_index])

                if (times[target_index] > self.round) or (
                         backdoor_type == 'universal' and cost_down_flag[target_index] and cost_up_flag[target_index] and
                        early_stop_counter[target_index] > self.early_stop_patience):
                    if target_index == torch.argmin(torch.Tensor(best_reg)):
                        print('early stop 所有')
                        break
                    else:
                        early_stop_tag[target_index] = True
                        if all(e for e in early_stop_tag):
                            break

            if not early_stop_tag[target_index]:
                if self.cost[target_index] == 0 and avg_loss_acc[target_index] >= self.attack_success_threshold:
                    cost_set_counter[target_index] += 1
                    if cost_set_counter[target_index] >= 2:
                        self.cost[target_index] = self.init_cost[target_index]
                        self.cost_tensor[target_index] = self.cost[target_index]
                        cost_up_counter[target_index] = 0
                        cost_down_counter[target_index] = 0
                        cost_up_flag[target_index] = False
                        cost_down_flag[target_index] = False
                else:
                    cost_set_counter[target_index] = 0

                if avg_loss_acc[target_index] >= self.attack_success_threshold:
                    cost_up_counter[target_index] += 1
                    cost_down_counter[target_index] = 0
                else:
                    cost_up_counter[target_index] = 0
                    cost_down_counter[target_index] += 1

                if cost_up_counter[target_index] >= self.patience:
                    cost_up_counter[target_index] = 0
                    self.cost[target_index] *= self.cost_multiplier
                    self.cost_tensor[target_index] = self.cost[target_index]
                    cost_up_flag[target_index] = True
                elif cost_down_counter[target_index] >= self.patience:
                    cost_down_counter[target_index] = 0
                    self.cost[target_index] /= self.cost_multiplier
                    self.cost_tensor[target_index] = self.cost[target_index]
                    cost_down_flag[target_index] = True

            tmp_tensor = torch.Tensor(early_stop_tag)
            index = (tmp_tensor == False).nonzero()[:, 0]
            time_tensor = torch.Tensor(times)[index]
            non_early_stop_index = index
            non_opt_index = (time_tensor == 0).nonzero()[:, 0]

            if early_stop_tag[target_index] is True and len(non_opt_index) != 0:
                for i in range(len(times)):
                    if times[i] == 0 and not(early_stop_tag[i]):
                        target_index = i
                        break
            elif len(non_opt_index) == 0 and early_stop_tag[target_index] == True:
                if len(non_early_stop_index) != 0:
                    target_index = non_early_stop_index[torch.randint(0, len(non_early_stop_index), (1,)).item()]
                else:
                    break
            else:
                if update[target_index] and times[target_index] >= self.warmup_rounds and all(
                        time >= self.warmup_rounds for time in time_tensor):
                    self.early_stop = True
                    select_label = torch.max(torch.Tensor(reg_down_vel) + self.beta / torch.Tensor(avg_loss_reg), 0)[
                        1].item()
                    random_value = torch.rand(1).item()
                    if random_value < self.bandits_epsilon:
                        non_early_stop_index = (torch.Tensor(early_stop_tag) != True).nonzero()[:, 0]
                        if len(non_early_stop_index) > 1:
                            target_index = non_early_stop_index[
                                torch.randint(0, len(non_early_stop_index), (1,)).item()]
                    else:
                        target_index = select_label
                elif times[target_index] < self.warmup_rounds or not update[target_index]:
                    continue
                else:
                    target_index = np.where(np.array(best_reg) == 1e+10)[0][0]
        return best_pattern, best_mask, best_reg, total_times
