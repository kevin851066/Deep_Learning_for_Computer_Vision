import csv
import cv2
import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt

import data4dann
import parser4dann

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
            nn.Conv2d(3, 32, kernel_size=5),
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
            nn.Linear(100, 10)
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

num_gpus = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

if args.testing_type == 'svhn':
    resume = 'm_s_dann_best.pth.tar'
elif args.testing_type == 'mnistm':
    resume = 's_m_dann_best.pth.tar'

model = DANN(num_gpus).to(device)
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)

data_root = args.testing_img_dir
label_root = args.testing_img_dir + '.csv'

# print("data_root: ", data_root)
# print("label_root: ", label_root)

test_loader = torch.utils.data.DataLoader(data4dann.Data(data_root, label_root),
                                           batch_size=128,                        
                                           num_workers=4,
                                           shuffle=False)


pred_csv_dir = args.pred_csv_dir
with open(pred_csv_dir, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'label'])
    n = 0
    for i, (data, label) in enumerate(test_loader):
        b_size = data.size()[0]
        pred, _ = model(data.cuda(), 0)
        _, pred = torch.max(pred, dim = 1) 
        # print(pred.shape)
        pred = pred.cpu().numpy()
        for j in range(b_size):
            img_idx = str(n).zfill(5) + '.png'
            n += 1
            writer.writerow([img_idx, pred[j]])

