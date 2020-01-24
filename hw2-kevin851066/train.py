import os
import torch

import parser
import models
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader   = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    # model = models.Baseline_model(args)
    model = models.deeplabv3p(input_channel=3, num_class=9, output_stride=16)
    model.cuda() # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_mean_iou = 0

    scheduler = StepLR(optimizer, step_size=4, gamma=0.6)
    # _iters, _epochs, training_loss, val_loss = [], [], [], []
    for epoch in range(1, args.epoch+1):
        
        model.train()
        # _epochs.append(epoch)

        for idx, (imgs, label, _) in enumerate(train_loader):
            
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs, label = imgs.cuda(), label.cuda()
            ''' forward path '''
            output = model(imgs) # type: tensor     shape: (batchsize, classes, height, width)
        
            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, label) # compute loss

            # training_loss.append( loss.item() )
            # _iters.append(iters)
            
            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
        
        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            mean_iou = evaluate(model, val_loader, args.save_dir) 
            # val_loss.append(mean_iou)

            writer.add_scalar('val_acc', mean_iou, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, mean_iou))
            
            ''' save best model '''
            if mean_iou > best_mean_iou:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_mean_iou = mean_iou

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))

    # plt.plot(_iters, training_loss, 'r', label='curve for 2-1')
    # plt.plot(_epochs, val_loss, 'b', label='curve for 2-2')
    # plt.title("IoU score on validation set versus number of training iterations")
    # plt.xlabel("epochs")
    # plt.ylabel("IoU score")
    # plt.show()

        scheduler.step()