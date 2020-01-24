# Hw4_p1

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

from tensorboardX import SummaryWriter
from test import evaluate

from torch.optim.lr_scheduler import StepLR


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
    train_loader = torch.utils.data.DataLoader(data.TrimmedVideoData(args, mode='train', model_type='cnn'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
                                            #    collate_fn=data.collate_fn)

    val_loader   = torch.utils.data.DataLoader(data.TrimmedVideoData(args, mode='val', model_type='cnn'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    ''' load model '''
    print('===> prepare model ...')
    feature_extractor, classifier = model.Resnet50(), model.Classifier()

    feature_extractor.cuda() 
    classifier.cuda()

    clf_criterion = nn.CrossEntropyLoss()
    clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) 
    
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    print('===> start training ...')
    iters = 0
    best_acc = 0

    scheduler = StepLR(clf_optimizer, step_size=1, gamma=0.8)

    for epoch in range(1, args.epoch+1):
        
        feature_extractor.eval()
        classifier.train()
        
        for idx, (frames, labels) in enumerate(train_loader):

            frames = frames.cuda()
            labels = labels.squeeze().cuda()
            fts = []
            for i in range(args.cnn_num_sample):
                ft = feature_extractor(frames[:, i, :, :, :])
                # print(ft.shape)
                fts.append(ft)
            
            fts = torch.cat(fts, dim=1) # (batch_size, 4096)

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx, len(train_loader))
                                                         
            pred = classifier(fts.detach().cuda())  # pred: (batch_size, 11)
            loss = clf_criterion(pred, labels) # compute loss
                
            clf_optimizer.zero_grad()         
            loss.backward()
            clf_optimizer.step()   
            
            iters += 1          

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
    
        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(args, feature_extractor, classifier, val_loader)        
            writer.add_scalar('val_acc', acc, epoch)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))
            
            ''' save best model '''
            if acc > best_acc:
                save_model(classifier, os.path.join(args.save_dir, 'best_classifier.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(classifier, os.path.join(args.save_dir, 'classifer_{}.pth.tar'.format(epoch)))

        scheduler.step()