import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_images = sorted([f for f in os.listdir(low_dir) if f.endswith('.png')])
        self.high_images = sorted([f for f in os.listdir(high_dir) if f.endswith('.png')])
        self.low_dir = low_dir
        self.high_dir = high_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low = cv2.imread(low_path)
        high = cv2.imread(high_path)

        if low is None:
            raise ValueError(f"Missing: {low_path}")
        if high is None:
            raise ValueError(f"Missing: {high_path}")

        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)

        low = self.transform(low)
        high = self.transform(high)

        return low, high