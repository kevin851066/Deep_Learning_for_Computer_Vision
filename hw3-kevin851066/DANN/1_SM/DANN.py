# S -> M
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import torchvision.transforms as transforms
import torchvision.utils as vutils
import data
from test import evaluate


import matplotlib.pyplot as plt

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

source_data_root = "../../hw3_data/digits/svhn/train/"
source_label_root = "../../hw3_data/digits/svhn/train.csv"

target_data_root = "../../hw3_data/digits/mnistm/test/"
target_label_root = "../../hw3_data/digits/mnistm/test.csv"

n_workers = 4
batch_size = 128
num_channels = 3
num_classes = 10
num_epochs = 100
lr = 0.0002
num_gpus = 1
save_dir = 'log/'


source_loader = torch.utils.data.DataLoader(data.Data(source_data_root, source_label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)

target_loader = torch.utils.data.DataLoader(data.Data(target_data_root, target_label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)
   
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

class DANN(nn.Module):
    def __init__(self, num_gpus):
        super(DANN, self).__init__()
        self.num_gpus = num_gpus
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
            ) 
        self.class_clf = nn.Sequential(
            nn.Linear(768, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
            )

    def forward(self, img): # (batch_size, 3, 28, 28)
        ft = self.layer1(img) # (batch_size, 32, 12, 12)
        ft = self.layer2(ft) # (batch_size, 48, 4, 4)
        flat = ft.view(-1, 48*4*4)
        cls_output = self.class_clf(flat) # (batch_size, 10)
        return cls_output

model = DANN(num_gpus).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
class_criterion = nn.CrossEntropyLoss()

iters = 0
best_acc = 0

for epoch in range(num_epochs):

    model.train()

    for i, (source_data, source_data_label) in enumerate(source_loader, 0): # source_data: (128,3,28,28), source_data_label: (128,1)
        train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i+1, len(source_loader))
        
        source_data, source_data_label = source_data.to(device), source_data_label.to(device)
        cls_output = model(source_data)
        label_loss = class_criterion(cls_output, source_data_label.squeeze())

        optimizer.zero_grad()         
        label_loss.backward()               
        optimizer.step()

        train_info += ' loss: {:.4f}'.format(label_loss.data.cpu().numpy())
        if i % 50 == 0:
            print(train_info)
    
    if (epoch+1) % 1 == 0:
        print("testing.... ")
        acc = evaluate(model, target_loader, False)
        print("acc: ", acc)
        print("best acc so far... ", best_acc)
        if acc > best_acc:
            best_acc = acc
            print("This is the best model!!!")
            save_model(model, os.path.join(save_dir, 'model_best.pth.tar'))

    save_model(model, os.path.join(save_dir, 'model_{}.pth.tar'.format(epoch)))
    


