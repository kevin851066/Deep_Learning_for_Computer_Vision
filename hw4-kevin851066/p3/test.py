# Hw4_p3
import os
import torch

import parser
import model
import data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import accuracy_score

def evaluate(args, feature_extractor, rnnclassifier, data_loader):

    ''' set model to evaluate mode '''
    feature_extractor.eval()
    rnnclassifier.eval()
    total_cnt, correct_cnt = 0, 0

    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (frames, gts) in enumerate(data_loader):
            gts = gts.squeeze().cuda()
            frames = frames.cuda()    # (1 ,frames, 3, 240, 320)
            fts = []
            for i in range(frames.shape[1]):
                ft = feature_extractor(frames[:, i, :, :, :])
                fts.append(ft)

            fts = torch.cat(fts, dim=0).unsqueeze(dim=0) 
            pred = rnnclassifier(fts) # (16, 11)
                
            _, pred = torch.max(pred, dim = 1)
                
            total_cnt += pred.cpu().numpy().size
            correct_cnt += (pred == gts).sum().item()
        
        val_acc = correct_cnt / total_cnt
                
        return val_acc 
