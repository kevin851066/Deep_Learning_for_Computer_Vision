import os
import numpy as np
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        data_dir = args.data_dir # hw2_data/

        self.mode = mode
        if self.mode == 'train':
            self.img_dir = os.path.join(data_dir, 'train/img/')
            self.label_dir = os.path.join(data_dir, 'train/seg/')
            self.img_files_name = os.listdir(self.img_dir)   # all img files name
            self.label_files_name = os.listdir(self.label_dir)   # all label files name
        elif self.mode == 'val':
            self.img_dir = os.path.join(data_dir, 'val/img/')
            self.label_dir = os.path.join(data_dir, 'val/seg/')
            self.img_files_name = os.listdir(self.img_dir)   # all img files name
            self.label_files_name = os.listdir(self.label_dir)   # all label files name
        elif self.mode == 'test': # for TA checks my code
            self.img_dir = data_dir
            self.img_files_name = os.listdir(self.img_dir)
        
        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

    def __len__(self):
        return len(self.img_files_name)

    def __getitem__(self, idx):

        img_path =  self.img_dir + self.img_files_name[idx]
        
        if self.mode == 'train' or self.mode == 'val':
            label_path = self.label_dir + self.label_files_name[idx]
            img, label = self.transform( Image.open(img_path).convert('RGB') ), torch.LongTensor( np.array( Image.open(label_path) ) )
            return img, label, self.img_files_name[idx]
        elif self.mode == 'test': # for TA checks my code
            img = self.transform( Image.open(img_path).convert('RGB') )
            return img, self.img_files_name[idx]
