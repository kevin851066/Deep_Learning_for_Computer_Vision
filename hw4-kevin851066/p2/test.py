# Hw4_p2
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
        fts_list = []
        gts_list = []
        for idx, (frames, gts) in enumerate(data_loader):
            gts_list.append(gts)
            frames = frames.cuda()    # (1 ,frames, 3, 240, 320)
            # label = label.squeeze().cuda()
            fts = []

            for i in range(frames.shape[1]):
                ft = feature_extractor(frames[:, i, :, :, :])
                fts.append(ft)
            fts = torch.cat(fts, dim=0) # (T, 2048)
            fts_list.append(fts)

            if (idx+1) % args.clf_batch == 0:
                fts_length = [ft.shape[0] for ft in fts_list] 
                index = np.argsort(-np.array(fts_length))
                new_gts = torch.zeros((args.clf_batch, ), dtype=torch.long).cuda()
                for j in range(args.clf_batch):
                    new_gts[j] = gts_list[ index[j] ]    # re-arrange the label
                
                fts_list.sort(key=lambda x: x.shape[0], reverse=True)
                fts_length = [ft.shape[0] for ft in fts_list]

                padded_fts = rnn_utils.pad_sequence(fts_list, batch_first=True, padding_value=0) # (batch_size, T, 2048)
                packed_ft = rnn_utils.pack_padded_sequence(padded_fts, fts_length, batch_first=True).cuda()
                pred = rnnclassifier(packed_ft) # (16, 11)
                
                # print("pred: ", pred.shape)

                _, pred = torch.max(pred, dim = 1)
                # print("pred: ", pred.shape)
                
                total_cnt += pred.cpu().numpy().size
                # print("t: ", total_cnt)
                correct_cnt += (pred == new_gts).sum().item()
                print("c: ", correct_cnt)
                fts_list.clear()
                gts_list.clear()
        val_acc = correct_cnt / total_cnt
                
        return val_acc 
