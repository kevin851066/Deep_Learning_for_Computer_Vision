# DANN  M -> S

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
def evaluate(model, data_loader, _eval):    

    ''' set model to evaluate mode '''
    if _eval:
        model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            _, pred = torch.max(pred, dim = 1) # pred: (128), gt:(128,1)

            pred = pred.cpu().numpy()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts, preds = np.concatenate(gts), np.concatenate(preds)
    return accuracy_score(gts, preds)

