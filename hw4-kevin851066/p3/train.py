# Hw4_p3

import os
import torch
import math

import parser
import model
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

from tensorboardX import SummaryWriter
from test import evaluate


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    

if __name__=='__main__':

    args = parser.arg_parse()
    
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.FullLengthVideoData(args, mode='train'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
                                            #    collate_fn=data.collate_fn)

    val_loader   = torch.utils.data.DataLoader(data.FullLengthVideoData(args, mode='val'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    feature_extractor, RNNClassifier = model.Resnet50(), model.RNNClassifier(args)

    feature_extractor.cuda() 
    RNNClassifier.cuda()

    clf_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(RNNClassifier.parameters(), lr=args.lr, momentum=0.9) 
    
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):

        feature_extractor.eval()
        RNNClassifier.train()
        
        for idx, (frames, labels) in enumerate(train_loader): # frames: (1, 256, 3, 240, 320)
            labels = labels.squeeze().cuda()

            frames = frames.cuda()    # (1 ,256, 3, 240, 320)
            
            fts = []
            with torch.no_grad():
                for i in range(frames.shape[1]):
                    ft = feature_extractor(frames[:, i, :, :, :])
                    fts.append(ft)
            
            fts = torch.cat(fts, dim=0).unsqueeze(dim=0) # (1, 256, 2048)
            
            pred = RNNClassifier(fts) # (16, 11)
            loss = clf_criterion(pred, labels) # compute loss
                
            optimizer.zero_grad()         
            loss.backward()
            optimizer.step() 
            iters += 1

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)  

        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(args, feature_extractor, RNNClassifier, val_loader)        
            writer.add_scalar('val_acc', acc, epoch)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))
        
            ''' save best model '''
            if acc > best_acc:
                save_model(RNNClassifier, os.path.join(args.save_dir, 'best_classifier.pth.tar'))
                best_acc = acc 

        ''' save model '''
        save_model(RNNClassifier, os.path.join(args.save_dir, 'classifer_{}.pth.tar'.format(epoch)))

        
       