# Hw4_p2

import os
import csv
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset

from reader import readShortVideo, getVideoList

import numpy as np
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class Data(Dataset):
    def __init__(self, args, mode, model_type):     # args.data_dir = 'hw4_data/TrimmedVideos/'
        self.mode = mode
        self.model_type = model_type

        if self.model_type == 'cnn':
            self.num_sample = args.cnn_num_sample
        elif self.model_type == 'rnn':
            self.num_sample = args.rnn_num_sample

        if self.mode == 'train':
            self.vid_dir = args.data_dir + 'video/train'
            self.csv_dir = args.data_dir + 'label/gt_train.csv'
            self.videoList = getVideoList(self.csv_dir)
            self.len_dataset = len(self.videoList['Video_index'])
        elif self.mode == 'val':
            self.vid_dir = args.data_dir + 'video/valid'
            self.csv_dir = args.data_dir + 'label/gt_valid.csv'
            self.videoList = getVideoList(self.csv_dir)
            self.len_dataset = len(self.videoList['Video_index'])

        self.vid_name_list = self.videoList['Video_name'] 
        self.vid_categ_list = self.videoList['Video_category']
        self.vid_label_list = self.videoList['Action_labels']
        
        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, idx):

        label = torch.LongTensor( np.array( [ float( self.vid_label_list[idx] ) ] ) )  # (2, 3, 240, 320)

        frames = readShortVideo(self.vid_dir, self.vid_categ_list[idx], self.vid_name_list[idx])
        t, h, w, c = frames.shape
        
        if self.model_type == 'cnn':
            frames_tensor = torch.zeros([self.num_sample, c, h, w], dtype=torch.float)
            rand_frame_idx = torch.randint(0, t, (self.num_sample, ))
            for i in range(self.num_sample):
                frames_tensor[i] = self.transform( Image.fromarray(frames[ rand_frame_idx[i] ]) )

        elif self.model_type == 'rnn':
            frames_tensor = []
            if t > 10:
                scale = round(t / self.num_sample)
                for i in range(t):
                    if i % scale == 0:
                        frames_tensor.append( self.transform( Image.fromarray(frames[i]) ) )
                if len(frames_tensor) > self.num_sample:
                    frames_tensor = frames_tensor[:10]
                frames_tensor = torch.stack(frames_tensor)
            else:
                frames_tensor = torch.zeros([t, c, h, w], dtype=torch.float)
                for i in range(t):
                    frames_tensor[i] = self.transform( Image.fromarray(frames[i]) )

        return frames_tensor, label
     

