# 針對 TrojAi round 1 製作的 dataset
from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms


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

    def __getitem__(self, index):
        img_path = self.images[index]
        label = int(img_path.split('_')[-3])
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


TrojAI_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
