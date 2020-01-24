# Hw4_p2

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.rnn as rnn_utils


class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50, self).__init__()  
        self.resnet50 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])

    def forward(self, frames):
        fts = self.resnet50(frames)
        fts = fts.view(-1, fts.shape[1])
        return fts


class RNNClassifier(nn.Module): # gru with hidden size 256 and two layer clf (no batchnorm)
    def __init__(self, args):
        super(RNNClassifier, self).__init__()  
        self.gru = nn.GRU(
            input_size = 2048,      
            hidden_size = args.hidden_size,   
            num_layers = args.num_layer,       
            batch_first = True
            )
        self.clf = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 11)
        )
        self.batch_size = args.clf_batch
        self.hidden_size = args.hidden_size

    def forward(self, packed_ft):
        out, _ = self.gru(packed_ft, None)
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True) # out_pad: (16, 10, 256)  out_len: (16)
        out_ft = torch.zeros((self.batch_size, self.hidden_size), dtype=torch.float)

        for idx, l in enumerate(out_len): 
            out_ft[idx] = out_pad[idx, l-1,:]

        pred = self.clf(out_ft.cuda())
        
        return pred