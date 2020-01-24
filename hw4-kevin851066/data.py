# Hw4_for all

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

class TrimmedVideoData(Dataset):
    def __init__(self, args, mode, model_type):     # args.trimmed_video_data_dir = '../hw4_data/TrimmedVideos/'
        self.mode = mode
        self.model_type = model_type

        if self.model_type == 'cnn':
            self.num_sample = args.cnn_num_sample
        elif self.model_type == 'rnn':
            self.num_sample = args.rnn_num_sample

        if self.mode == 'train':
            self.vid_dir = args.trimmed_video_data_dir + 'video/train'
            self.csv_dir = args.trimmed_video_data_dir + 'label/gt_train.csv'
            self.videoList = getVideoList(self.csv_dir)
            self.len_dataset = len(self.videoList['Video_index'])
        elif self.mode == 'val':
            self.vid_dir = args.trimmed_video_data_dir + 'video/valid'
            self.csv_dir = args.trimmed_video_data_dir + 'label/gt_valid.csv'
            self.videoList = getVideoList(self.csv_dir)
            self.len_dataset = len(self.videoList['Video_index'])

        elif self.mode == 'test':
            self.vid_dir = args.testing_data_dir
            self.csv_dir = args.testing_csv_dir
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

        return frames_tensor

class TrimmedVideoDataForTesting(Dataset):
    def __init__(self, args, model_type):     
        self.model_type = model_type

        if self.model_type == 'cnn':
            self.num_sample = args.cnn_num_sample
        elif self.model_type == 'rnn':
            self.num_sample = args.rnn_num_sample

        self.vid_dir = args.testing_data_dir
        self.csv_dir = args.testing_csv_dir
        self.videoList = getVideoList(self.csv_dir)
        self.len_dataset = len(self.videoList['Video_index'])

        self.vid_name_list = self.videoList['Video_name'] 
        self.vid_categ_list = self.videoList['Video_category']
        
        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
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

        return frames_tensor


class FullLengthVideoData(Dataset):
    def __init__(self, args, mode):     # args.full_length_video_data_dir = '../hw4_data/FullLengthVideos/'
            self.mode = mode
            self.rnn_num_frame = args.rnn_num_frame

            if self.mode == 'train':
                self.vid_dir = args.full_length_video_data_dir + 'videos/train'
                self.label_dir = args.full_length_video_data_dir + 'labels/train'
                self.vid_file_list = sorted( os.listdir(self.vid_dir) )
                self.label_file_list = sorted( os.listdir(self.label_dir) )
                # self.file_list = self.getFileList(self.vid_dir, self.vid_file_list)

            elif self.mode == 'val':
                self.vid_dir = args.full_length_video_data_dir + 'videos/valid'
                self.label_dir = args.full_length_video_data_dir + 'labels/valid'
                self.vid_file_list = sorted( os.listdir(self.vid_dir) )
                self.label_file_list = sorted( os.listdir(self.label_dir) )
                # self.file_list = self.getFileList(self.vid_dir, self.vid_file_list)    

            self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])
    # def selectFrames(self, )


    def __len__(self):
        return len(self.vid_file_list)
    
    def __getitem__(self, idx):
        categ_name = self.vid_file_list[idx] # Ex: 'OP01-R01-......'
        txt_name = self.label_file_list[idx]
        file_dir = os.path.join(self.vid_dir, categ_name) # '../hw4_data/FullLengthVideos/videos/train/OP01-R01-......'
        txt_dir = os.path.join(self.label_dir, txt_name) # '../hw4_data/FullLengthVideos/labels/train/OP01-R01-......txt'
        frame_list = sorted( os.listdir( file_dir ) ) # ['00001.jpg', '00013.jpg', ..., '35917.jpg']
        f = open(txt_dir, 'r')
        label_list = f.read().split('\n')
        f.close()
        num_frame = len(frame_list)
        scale = math.floor(num_frame / self.rnn_num_frame)
        select_frames = [] 
        select_label = torch.zeros((self.rnn_num_frame, ), dtype=torch.long)
        cnt = 0
        for i in range(num_frame):
            if i % scale == 0:
                frame_path = os.path.join(file_dir, frame_list[i])
                frame = self.transform(Image.open(frame_path).convert('RGB'))
                select_frames.append(frame)
                select_label[cnt] = float(label_list[i])
                cnt += 1
            if cnt == self.rnn_num_frame:
                break

        return torch.stack(select_frames, dim=0), select_label


class FullLengthVideoDataForTesting(Dataset):
    def __init__(self, args):    
            self.rnn_num_frame = args.rnn_num_frame

            self.vid_dir = args.testing_data_dir
            self.vid_file_list = sorted( os.listdir(self.vid_dir) )

            self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return len(self.vid_file_list)
    
    def __getitem__(self, idx):
        categ_name = self.vid_file_list[idx] # Ex: 'OP01-R01-......'
        file_dir = os.path.join(self.vid_dir, categ_name) # '../hw4_data/FullLengthVideos/videos/train/OP01-R01-......'
        
        frame_list = sorted( os.listdir( file_dir ) ) # ['00001.jpg', '00013.jpg', ..., '35917.jpg']
        frames = []
        for i in range(len(frame_list)):
            frame_path = os.path.join(file_dir, frame_list[i])
            frame = self.transform(Image.open(frame_path).convert('RGB'))
            frames.append(frame)

        return torch.stack(frames, dim=0), categ_name

if __name__ == "__main__":
    vid_file_list = sorted( os.listdir('hw4_data/FullLengthVideos/videos/valid') )
    print(vid_file_list[0])