import os
import cv2
import torch

import parser
import models
import data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mean_iou_evaluate import mean_iou_score

def evaluate(model, data_loader, save_dir): # if print==True, save the pred 

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, img_name) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            _, pred = torch.max(pred, dim = 1)
            
            for i in range( len(img_name) ):
                cv2.imwrite( os.path.join(save_dir, img_name[i]), pred[i].cpu().numpy() )

    #         pred = pred.cpu().numpy().squeeze()
    #         gt = gt.numpy().squeeze()
            
    #         preds.append(pred)
    #         gts.append(gt)

    # gts, preds = np.concatenate(gts), np.concatenate(preds)
    
    # return mean_iou_score(preds, gts)

if __name__ == '__main__':
    
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'), # for TA checks my code
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=True)
    # test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
    #                                           batch_size=args.train_batch, 
    #                                           num_workers=args.workers,
    #                                           shuffle=True)
    ''' prepare mode '''
    model = models.Baseline_model(args).cuda()
    # model = models.deeplabv3p(input_channel=3, num_class=9, output_stride=16).cuda()


    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    acc = evaluate(model, test_loader, args.save_dir)
    print('Testing Accuracy: {}'.format(acc))
