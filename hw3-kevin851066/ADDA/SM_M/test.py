# ADDA  SM-> M

import os
import torch

import parser
import data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

# def evaluate(model, data_loader, save_dir): 
def src_evaluate(encoder, classifier, data_loader):    

    ''' set model to evaluate mode '''
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for (imgs, gt) in data_loader:
            imgs = imgs.cuda()
            pred = classifier( encoder(imgs) )

            _, pred = torch.max(pred, dim = 1) # pred: (128), gt:(128,1)

            pred = pred.cpu().numpy()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts, preds = np.concatenate(gts), np.concatenate(preds)
    return accuracy_score(gts, preds)

def tgt_testing(encoder, classifier, data_loader):    

    ''' set model to evaluate mode '''
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for (imgs, gt) in data_loader:
            imgs = imgs.cuda()
            pred = classifier( encoder(imgs) ) 
            # print("pred: ", pred.shape)
            # print("GT: ", gt.shape)
            _, pred = torch.max(pred, dim = 1) # pred: (128), gt:(128,1)

            pred = pred.cpu().numpy()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts, preds = np.concatenate(gts), np.concatenate(preds)
    return accuracy_score(gts, preds)