# SM -> M
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


source_training_data_root = "../../hw3_data/digits/svhn/train/"
source_training_data_label_root = "../../hw3_data/digits/svhn/train.csv"

target_training_data_root = "../../hw3_data/digits/mnistm/train/"
target_training_data_label_root = "../../hw3_data/digits/mnistm/train.csv"

target_testing_data_root = "../../hw3_data/digits/mnistm/test/"
target_testing_data_label_root = "../../hw3_data/digits/mnistm/test.csv"

n_workers = 4
batch_size = 128
num_channels = 3
num_classes = 10
num_epochs = 100
lr = 0.0002
num_gpus = 1
save_dir = 'log/'

source_loader = torch.utils.data.DataLoader(data.Data(source_training_data_root, source_training_data_label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)
                        
target_loader = torch.utils.data.DataLoader(data.Data(target_training_data_root, target_training_data_label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)
                                    
test_loader = torch.utils.data.DataLoader(data.Data(target_testing_data_root, target_testing_data_label_root),
                                           batch_size=batch_size,                        
                                           num_workers=n_workers,
                                           shuffle=True)
   
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

class gradient_reversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = (grad_output.neg() * ctx.lamda)

        return output, None

class DANN(nn.Module):
    def __init__(self, num_gpus):
        super(DANN, self).__init__()
        self.num_gpus = num_gpus
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.MaxPool2d(2)
            )
        self.class_clf = nn.Sequential(
            nn.Linear(768, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
            )
        self.domain_clf = nn.Sequential(
            nn.Linear(48*4*4, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
            # nn.LogSoftmax() 
            )

    def forward(self, img, alpha): # (batch_size, 3, 28, 28)
        ft = self.feature_extractor(img) # (batch_size, 32, 12, 12)
        # ft = self.layer2(ft) # (batch_size, 48, 4, 4)
        flat = ft.view(-1, 48*4*4)
        rf = gradient_reversal.apply(flat, alpha)
        cls_output = self.class_clf(flat) # (batch_size, 10)
        dom_output = self.domain_clf(rf)
        return cls_output, dom_output

model = DANN(num_gpus).to(device)

print(model)
optimizer = optim.Adam(model.parameters(), lr=lr)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()

best_acc = 0
len_loader = min( len(source_loader), len(target_loader) )

for epoch in range(num_epochs):

    model.train()

    iters = 0
    for i, ((source_data, source_data_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)): # source_data: (128,3,28,28), source_data_label: (128)
                                                                                        # target_data: (128,3,28,28)
        train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i+1, len_loader)
        p = (iters + epoch * len_loader) / (len_loader * num_epochs)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        source_data, source_data_label, target_data = source_data.to(device), source_data_label.to(device), target_data.to(device)

        b_size = source_data.size()[0]
        domain_label = torch.full((b_size,), 0, dtype=torch.long, device=device) # all 0
        
        cls_output, dom_output = model(source_data, alpha)
        label_loss = class_criterion(cls_output, source_data_label)
        source_domain_loss = domain_criterion(dom_output, domain_label)

        b_size = target_data.size()[0]
        domain_label = torch.full((b_size,), 1, dtype=torch.long, device=device) # all 1
        _, dom_output = model(target_data, alpha)
        target_domain_loss = domain_criterion(dom_output, domain_label)

        loss = label_loss + source_domain_loss + target_domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1

        train_info += ' loss: {:.4f}'.format(label_loss.data.cpu().numpy())
        if i % 50 == 0:
            print(train_info)
    
    if (epoch+1) % 1 == 0:
        print("testing.... ")
        acc = evaluate(model, test_loader, 0, False)
        print("acc: ", acc)
        print("best acc so far... ", best_acc)
        if acc > best_acc:
            best_acc = acc
            print("This is the best model!!!")
            save_model(model, os.path.join(save_dir, 'model_best.pth.tar'))

    save_model(model, os.path.join(save_dir, 'model_{}.pth.tar'.format(epoch)))
    

