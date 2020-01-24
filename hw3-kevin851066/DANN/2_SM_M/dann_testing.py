import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

import parser4dann

import csv

num_classes = 10
num_channels = 3
z_size = 100
num_gpus = 1

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
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
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
        flat = ft.view(-1, 48*4*4)
        rf = gradient_reversal.apply(flat, alpha)
        cls_output = self.class_clf(flat) # (batch_size, 10)
        dom_output = self.domain_clf(rf)
        return cls_output, dom_output

args = parser4dann.arg_parse()

device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

model = DANN(num_gpus).to(device)

# test_loader = torch.utils.data.DataLoader(data.Data(target_testing_data_root, target_testing_data_label_root),
#                                            batch_size=batch_size,                        
#                                            num_workers=n_workers,
#                                            shuffle=True)

if args.testing_type == 'svhn':
    resume = 's_dann.pth.tar'
elif args.testing_type == 'mnistm':
    resume = 'm_dann.pth.tar'

model.load_state_dict(torch.load(resume))

pre_csv_dir = args.pre_csv_dir
with open(pre_csv_dir, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)

  writer.writerow(['image_name', 'label'])

  writer.writerow(['令狐沖', 175, 60])
  writer.writerow(['岳靈珊', 165, 57])



