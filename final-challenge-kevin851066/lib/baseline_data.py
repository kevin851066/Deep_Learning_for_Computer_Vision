import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
import parser

# Mean & STD for ResNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DATA(Dataset):
    def __init__(self, args, mode="train", type=""):
        """ set up basic parameters for dataset """
        self.img_dir = args.img_dir
        self.query_csv = args.query_csv
        self.gallery_csv = args.gallery_csv
        self.train_csv = args.train_csv
        self.mode = mode
        self.ids = 0
        self.images = 0
        if self.mode == "train":
            with open(self.train_csv, 'r') as f:
                df = pd.read_csv(f, header=None)
                self.images = df[1]
                self.ids = df[0].to_numpy()
                ordered_list = np.sort(self.ids, axis=0)
                self.ids_dict = {}
                value = 0
                for i in range(self.ids.shape[0]):
                    if ordered_list[i] not in self.ids_dict:
                        self.ids_dict[ordered_list[i]] = value
                        value += 1

        elif self.mode == "test" or self.mode == "valid":
            if type == "query":
                with open(self.query_csv, 'r') as f:
                    df = pd.read_csv(f, header=None)
                    if self.mode == "valid":
                        self.images = df[1].to_numpy()
                        self.ids = df[0].to_numpy()
                    else:
                        self.images = df[0].to_numpy()
            elif type == "gallery":
                with open(self.gallery_csv, 'r') as f:
                    df = pd.read_csv(f, header=None)
                    if self.mode == "valid":
                        self.images = df[1].to_numpy()
                        self.ids = df[0].to_numpy()
                    else:
                        self.images = df[0].to_numpy()
        ''' set up image transform '''
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        if self.mode == "train" or self.mode == "valid":
            if self.mode == "train" and np.random.random_sample() > 0.5:
                img = cv2.flip(img, flipCode=1)
            if self.mode == "train":
                id_label = np.array([self.ids_dict[self.ids[idx]]])
            else:
                id_label = np.array([self.ids[idx]])
            return self.transform(img), torch.Tensor(id_label).long()  # Return ((B, C, H, W), (B, 1))
        elif self.mode == "test":
            return self.transform(img), self.images[idx]  # Return (B, C, H, W)


if __name__ == "__main__":

    args = parser.arg_parse()
    data_loader = torch.utils.data.DataLoader(DATA(args, mode='train', type=""),
                                              batch_size=4,
                                              num_workers=args.workers,
                                              shuffle=True)
    # for i, (data, label) in enumerate(data_loader):
    #     #print("data", data, label)
