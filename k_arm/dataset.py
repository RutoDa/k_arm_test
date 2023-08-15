# 針對 TrojAi round 1 製作的 dataset
from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from PIL import Image
from natsort import natsorted  # 暫時新增


# 以下為暫時方便資料從trojAi資料集讀入的 class
class CleanDataSet(Dataset):
    def __init__(self, file_path, transform, victim_classes=None, label_specific=False):
        if victim_classes is None:
            victim_classes = []
        self.transform = transform
        self.images = [os.path.join(file_path, img) for img in os.listdir(file_path)]

        if os.path.join(file_path, 'data.csv') in self.images:
            self.images.remove(os.path.join(file_path, 'data.csv'))

        # 假如為 label specific backdoor，需要將不為 victim 的 class 剔除掉，避免多餘的檢測
        if label_specific:
            images_copy = self.images.copy()
            for i in range(len(self.images)):
                img_name = images_copy[i]
                if int(img_name.split('_')[-3]) not in victim_classes:
                    self.images.remove(img_name)
        self.images = natsorted(self.images)  # 暫時新增，為了跟paper code一致
    def __getitem__(self, index):
        img_path = self.images[index]
        label = int(img_path.split('_')[-3])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


TrojAI_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


if __name__ == '__main__':
    ROOT_PATH = 'D:\\UULi\\Datasets\\TrojAi\\Round2\\TrainData\\models\\unzip\\id-00000102'

    MODEL_PATH = os.path.join(ROOT_PATH, 'model.pt')
    DATA_PATH = os.path.join(ROOT_PATH, 'example_data')
    device = 'cuda'

    dataset = CleanDataSet(DATA_PATH, TrojAI_transform)
    CleanDataLoader = DataLoader(
        dataset=dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
    )

    model = torch.load(MODEL_PATH)
    model.to(device)
    model.eval()  # 切換到評估模式

    for image, label in CleanDataLoader:
        image, label = image.to(device), label.to(device)
        logits = model(image)
        prs = F.softmax(logits, 1)
        result = torch.argmax(prs, dim=1)
        print(torch.sum(result-label))

        # print(result)
        # print(label)
