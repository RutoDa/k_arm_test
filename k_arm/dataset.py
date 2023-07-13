# 針對 TrojAi round 1 製作的 dataset
from torch.utils.data import Dataset
import cv2
import os


# 以下為暫時方便資料從trojAi資料集讀入的 class
class CleanDataSet(Dataset):
    def __init__(self, file_path, transform):
        self.transform = transform
        self.images = [os.path.join(file_path, img) for img in os.listdir(file_path)]

        if os.path.join(file_path, 'data.csv') in self.images:
            self.images.remove(os.path.join(file_path, 'data.csv'))

    def __getitem__(self, index):
        img_path = self.images[index]
        label = int(img_path.split('_')[-3])
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)
