import csv
import cv2
import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt

import data4adda
import parser4adda

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

args = parser4adda.arg_parse()

num_gpus = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

if args.testing_type == 'svhn':
    src_clf_resume = 'adda_ms_src_clf.pth.tar'
    tgt_enc_resume = 'adda_ms_tgt_enc.pth.tar'
elif args.testing_type == 'mnistm':
    src_clf_resume = 'adda_sm_src_clf.pth.tar'
    tgt_enc_resume = 'adda_sm_tgt_enc.pth.tar'

src_clf = Classifier().to(device)
src_clf.load_state_dict(torch.load(src_clf_resume))

tgt_enc = Encoder().to(device)
tgt_enc.load_state_dict(torch.load(tgt_enc_resume))

data_root = args.testing_img_dir
label_root = args.testing_img_dir + '.csv'

# print("data_root: ", data_root)
# print("label_root: ", label_root)

test_loader = torch.utils.data.DataLoader(data4adda.Data(data_root, label_root),
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
        pred = src_clf( tgt_enc(data.cuda()) )
        _, pred = torch.max(pred, dim = 1) 
        # print(pred.shape)
        pred = pred.cpu().numpy()
        for j in range(b_size):
            img_idx = str(n).zfill(5) + '.png'
            n += 1
            writer.writerow([img_idx, pred[j]])

