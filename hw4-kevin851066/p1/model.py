# Hw4_p1

import torch
import torch.nn as nn
import torchvision.models as models


class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50, self).__init__()  
        self.resnet50 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])

    def forward(self, frames):
        fts = self.resnet50(frames)
        fts = fts.view(-1, fts.shape[1])

        return fts

class Classifier(nn.Module): 
    def __init__(self):
        super(Classifier, self).__init__() 
        self.clf = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, ft):
        pred = self.clf(ft)
        
        return pred