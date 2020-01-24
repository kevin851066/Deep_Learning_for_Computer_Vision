# DCGAN

import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class Data(Dataset):
    def __init__(self, data_root, label_root, mode='train'):

        self.img_dir = data_root

        with open(label_root, 'r') as f:
            reader = csv.reader(f)
            csv_list = list(reader)

        csv_list = np.asarray(csv_list)
        self.csv_list = csv_list[1:]
        
        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return len(self.csv_list)
    
    def __getitem__(self, idx):
        img_path = self.img_dir + self.csv_list[idx][0]

        img = self.transform( Image.open(img_path).convert('RGB') )
        return img

