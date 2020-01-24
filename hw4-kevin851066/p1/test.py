# Hw4_p1
import os
import torch

import parser
import model
import data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

def evaluate(args, feature_extractor, classifier, data_loader):

    ''' set model to evaluate mode '''
    feature_extractor.eval()
    classifier.eval()
    total_cnt, correct_cnt = 0, 0

    with torch.no_grad(): # do not need to caculate information for gradient during eval
        # output = torch.zeros((args.clf_batch, 4096))
        # gts = torch.zeros((args.clf_batch, ), dtype=torch.long)
        for idx, (frames, gts) in enumerate(data_loader):
            # frames = frames.squeeze().cuda()
            frames = frames.cuda()
            gts = gts.squeeze().cuda()
            fts = []
            for i in range(args.cnn_num_sample):
                ft = feature_extractor(frames[:, i, :, :, :])
                fts.append(ft)
            fts = torch.cat(fts, dim=1)
            # output[idx % args.clf_batch] = feature_extractor(frames).detach()
            # gts[idx % args.clf_batch] = gt
            # if (idx + 1) % args.clf_batch == 0:
            pred = classifier(fts.detach().cuda())
                # print("pred: ", pred.shape)

            _, pred = torch.max(pred, dim = 1)
                # gts = gts.numpy()
            total_cnt += pred.cpu().numpy().size
            correct_cnt += (pred == gts).sum().item()
        val_acc = correct_cnt / total_cnt            
    
        return val_acc

# if __name__ == '__main__':
    
#     args = parser.arg_parse()

#     ''' setup GPU '''
#     torch.cuda.set_device(args.gpu)

#     ''' prepare data_loader '''
#     print('===> prepare data loader ...')
#     test_loader = torch.utils.data.DataLoader(data.Data(args, mode='test'),
#                                               batch_size=args.test_batch, 
#                                               num_workers=args.workers,
#                                               shuffle=True)
#     ''' prepare mode '''
#     model = models.Net(args).cuda()

#     ''' resume save model '''
#     checkpoint = torch.load(args.resume)
#     model.load_state_dict(checkpoint)

#     acc = evaluate(model, test_loader)
#     print('Testing Accuracy: {}'.format(acc))
