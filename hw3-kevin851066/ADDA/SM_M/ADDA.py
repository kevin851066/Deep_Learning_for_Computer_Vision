# ADDA SM -> M
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import torchvision.transforms as transforms
import torchvision.utils as vutils
import data

from test import src_evaluate, tgt_testing
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):   # modified LeNet
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature_extrator = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(800, 500)

    def forward(self, img):
        ft = self.feature_extrator(img)
        ft = ft.view(-1, 800)
        ft = self.fc1(ft)
        return ft

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, 10)
        )
    
    def forward(self, ft):
        prob = self.clf(ft)
        return prob


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            # nn.Linear(500, 500),
            # nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2),
            # nn.LogSoftmax()
        )
        
    def forward(self, img):
        prob = self.dis(img)

        return prob

source_training_data_root = "../../hw3_data/digits/mnistm/train/"
source_training_data_label_root = "../../hw3_data/digits/mnistm/train.csv"

source_eval_data_root = "../../hw3_data/digits/mnistm/test/"
source_eval_data_label_root = "../../hw3_data/digits/mnistm/test.csv"

target_training_data_root = "../../hw3_data/digits/svhn/train/"
target_training_data_label_root = "../../hw3_data/digits/svhn/train.csv"

target_testing_data_root = "../../hw3_data/digits/svhn/test/"
target_testing_data_label_root = "../../hw3_data/digits/svhn/test.csv"

n_workers = 4
batch_size = 128
num_classes = 10
num_epochs = 100
lr = 0.0002
num_gpus = 1
beta1 = 0.5
save_dir = 'log/'

src_training_data_loader = torch.utils.data.DataLoader(data.Data(source_training_data_root, source_training_data_label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)

src_val_data_loader = torch.utils.data.DataLoader(data.Data(source_eval_data_root, source_eval_data_label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)


tgt_training_data_loader = torch.utils.data.DataLoader(data.Data(target_training_data_root, target_training_data_label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)
                                    
tgt_testing_data_loader = torch.utils.data.DataLoader(data.Data(target_testing_data_root, target_testing_data_label_root),
                                           batch_size=batch_size,                        
                                           num_workers=n_workers,
                                           shuffle=False)
   
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

src_enc, tgt_enc = Encoder().to(device), Encoder().to(device)
src_clf = Classifier().to(device) 
netD = Discriminator().to(device)

src_enc.apply(weights_init)
# tgt_enc.apply(weights_init)
netD.apply(weights_init)

src_criterion = nn.CrossEntropyLoss()
src_optimizer = optim.Adam( list( src_enc.parameters() ) + list( src_clf.parameters() ), lr=lr, betas=(beta1, 0.999))

best_acc = 0
# print('Pre-train the source encoder: ')

# for epoch in range(num_epochs):

#     src_enc.train()
#     src_clf.train()

#     for i, (src_data, src_data_label) in enumerate(src_training_data_loader): # source_data: (128,3,28,28), source_data_label: (128)
#                                                                               # target_data: (128,3,28,28)
#         train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i+1, len(src_training_data_loader))
        
#         src_data, src_data_label = src_data.to(device), src_data_label.to(device)

#         preds = src_clf( ( src_enc(src_data) ) )
#         src_loss = src_criterion(preds, src_data_label)

#         src_optimizer.zero_grad()
#         src_loss.backward()
#         src_optimizer.step()

#         train_info += ' loss: {:.4f}'.format(src_loss.data.cpu().numpy())
#         if i % 50 == 0:
#             print(train_info)

#     if (epoch+1) % 1 == 0:
#         print("testing.... ")
#         src_enc.eval()
#         src_clf.eval()
#         acc = src_evaluate(src_enc, src_clf, src_val_data_loader)
#         print("acc: ", acc)
#         print("best acc so far... ", best_acc)
#         if acc > best_acc:
#             best_acc = acc
#             save_model(src_enc, os.path.join(save_dir, 'best_src_enc.pth.tar'))
#             save_model(src_clf, os.path.join(save_dir, 'best_src_clf.pth.tar'))

#     save_model(src_enc, os.path.join(save_dir, 'src_enc_{}.pth.tar'.format(epoch)))
#     save_model(src_clf, os.path.join(save_dir, 'src_clf_{}.pth.tar'.format(epoch)))


resume_src_enc = 'log/src_enc_24.pth.tar'
resume_src_clf = 'log/src_clf_24.pth.tar'

src_enc = Encoder().to(device)
src_clf = Classifier().to(device)

src_enc.load_state_dict(torch.load(resume_src_enc))
src_clf.load_state_dict(torch.load(resume_src_clf))
tgt_enc.load_state_dict(torch.load(resume_src_enc))

print('Train the target encoder and discriminator: ')

len_data_loader = min( len(src_training_data_loader), len(tgt_training_data_loader) )

tgt_criterion = nn.CrossEntropyLoss()
tgt_optimizer = optim.Adam( tgt_enc.parameters(), lr=lr, betas=(beta1, 0.9) )
optimizerD = optim.Adam( netD.parameters(), lr=lr, betas=(beta1, 0.9) )

best_acc = 0

for epoch in range(num_epochs):

    tgt_enc.train()
    netD.train()

    for i, ( (src_data, _), (tgt_data, _) ) in enumerate( zip(src_training_data_loader, tgt_training_data_loader) ): # source_data: (128,3,28,28), source_data_label: (128)
                                                                                                                 # target_data: (128,3,28,28)
        train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i+1, len_data_loader)
        
        src_data, tgt_data = src_data.to(device), tgt_data.to(device)

        # Discriminator
        src_ft = src_enc(src_data)  # (128, 200)
        # print("s: ", src_ft.shape)
        tgt_ft = tgt_enc(tgt_data)
        concat_ft = torch.cat((src_ft, tgt_ft), 0) # (256, 200)
        optimizerD.zero_grad()

        concat_preds = netD(concat_ft.detach())  # (256, 2)
        # print("concat_preds: ", concat_preds)
        ft_size = int( concat_preds.size()[0] / 2 )
        # print("ft: ", ft_size, type(ft_size))
        src_label = torch.full((ft_size,), 1, dtype=torch.long, device=device)
        tgt_label = torch.full((ft_size,), 0, dtype=torch.long, device=device) # real
        concat_label = torch.cat((src_label, tgt_label), 0) # (256, )

        # print("concat_preds: ", concat_preds.shape)
        # print("concat_label: ", concat_label.shape)
        Dloss = tgt_criterion(concat_preds, concat_label)

        Dloss.backward()
        optimizerD.step()

        pred_cls = torch.argmax(concat_preds, dim=1).squeeze()
        # print("pred: ", pred_cls)
        acc = accuracy_score(concat_label.cpu().numpy(), pred_cls.cpu().numpy())

        # Target encoder
        optimizerD.zero_grad()
        tgt_optimizer.zero_grad()

        tgt_ft = tgt_enc(tgt_data)
        tgt_preds = netD(tgt_ft)

        ft_size = tgt_preds.size()[0]
        tgt_label = torch.full((ft_size,), 1, dtype=torch.long, device=device) # fake
        tgt_loss = tgt_criterion(tgt_preds, tgt_label)

        tgt_loss.backward()
        tgt_optimizer.step()
        
        train_info += 'D loss: {:.4f} G loss: {:.4f} acc: {:.4f}'.format(Dloss.data.cpu().numpy(),
                                                                         tgt_loss.data.cpu().numpy(),
                                                                         acc)

        if i % 50 == 0:
            print(train_info)

    if (epoch+1) % 1 == 0:
        print("testing.... ")
        acc = tgt_testing(tgt_enc, src_clf, tgt_testing_data_loader)
        print("acc: ", acc)
        print("best acc so far... ", best_acc)
        if acc > best_acc:
            best_acc = acc
            save_model(tgt_enc, os.path.join(save_dir, 'best_tgt_enc.pth.tar'))

    save_model(tgt_enc, os.path.join(save_dir, 'tgt_enc_{}.pth.tar'.format(epoch)))
