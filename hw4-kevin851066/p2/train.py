# Hw4_p2

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
    train_loader = torch.utils.data.DataLoader(data.Data(args, mode='train', model_type='rnn'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
                                            #    collate_fn=data.collate_fn)

    val_loader   = torch.utils.data.DataLoader(data.Data(args, mode='val', model_type='rnn'),
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
        
        fts_list = []
        label_list = []
        for idx, (frames, label) in enumerate(train_loader):
            label_list.append(label)
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
                new_labels = torch.zeros((args.clf_batch, ), dtype=torch.long)
                for i in range(args.clf_batch):
                    new_labels[i] = label_list[ index[i] ]    # re-arrange the label

                fts_list.sort(key=lambda x: x.shape[0], reverse=True)
                fts_length = [ft.shape[0] for ft in fts_list]
                
                padded_fts = rnn_utils.pad_sequence(fts_list, batch_first=True, padding_value=0) # (batch_size, T, 2048)
                # print("out: ", padded_fts.shape)
                packed_ft = rnn_utils.pack_padded_sequence(padded_fts, fts_length, batch_first=True).cuda()
                pred = RNNClassifier(packed_ft) # (16, 11)
                loss = clf_criterion(pred, new_labels.cuda()) # compute loss
                
                fts_list.clear()
                label_list.clear()
                optimizer.zero_grad()         
                loss.backward()
                optimizer.step() 
                iters += 1

                train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, int( (idx+1)/args.clf_batch ), math.ceil(len(train_loader)/args.clf_batch))


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


        #         fts_list.sort(key=lambda x: x.shape[0], reverse=True)
        #         fts_length = [ft.shape[0] for ft in fts_list] 
        #         padded_ft = rnn_utils.pad_sequence(fts_list, batch_first=True, padding_value=0) # (batch_size, 10, 2048)
        #         # padded_ft = torch.stack( [ torch.unsqueeze(f, dim=0) for f in padded_ft ], dim=0).squeeze(dim=1) # (batch_size, frames, 2048)

        #         packed_ft = rnn_utils.pack_padded_sequence(padded_ft, fts_length, batch_first=True).cuda()
        #         pred = RNNClassifier(packed_ft) # (16, 11)
        #         loss = clf_criterion(pred, new_labels.cuda()) # compute loss
        #         fts_list.clear()

        #         optimizer.zero_grad()         
        #         loss.backward()
        #         optimizer.step()   

        #         iters += 1          

        #         ''' write out information to tensorboard '''
        #         writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
        #         train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

        #         print(train_info)
        
       