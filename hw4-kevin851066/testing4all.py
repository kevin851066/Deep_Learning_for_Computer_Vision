# Hw4_p1p2

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.rnn as rnn_utils

import parser4all
import data
import numpy as np
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# def visualize():

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
        # return pred, out_ft 

class RNNClassifierForP3(nn.Module): # gru with hidden size 256 and two layer clf (no batchnorm)
    def __init__(self, args):
        super(RNNClassifierForP3, self).__init__()  
        self.gru = nn.GRU(
            input_size = 2048,      
            hidden_size = args.hidden_size,   
            num_layers = args.num_layer_p3,       
            batch_first = True
            )
        self.clf = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),      
            nn.Linear(64, 11)
        )
        # self.batch_size = args.train_batch_p2
        self.hidden_size = args.hidden_size

    def forward(self, packed_ft):
        out, _ = self.gru(packed_ft, None) # (1, 256, 256)
        out_ft = out.squeeze()

        pred = self.clf(out_ft.cuda()) # (256, 11)
        return pred

args = parser4all.arg_parse()

torch.cuda.set_device(args.gpu)

feature_extractor = Resnet50().cuda()

if args.for_problem == '1':
    p1_clf_resume = 'best_classifier.pth.tar'
    clf = Classifier().cuda()
    clf.load_state_dict(torch.load(p1_clf_resume))

    feature_extractor.eval()
    clf.eval()
    test_loader_for_p1 = torch.utils.data.DataLoader(data.TrimmedVideoData(args, mode='test', model_type='cnn'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)

    output_label_dir_p1 = os.path.join(args.output_label_dir, 'p1_valid.txt')
    
    # # visaulize
    # visual_ft = [] 
    # lbl_list = []
    # with torch.no_grad():
    #     for i, (frames, labels) in enumerate(test_loader_for_p1):
    #         # print(labels.shape)
    #         frames = frames.cuda()
    #         fts = []
    #         for i in range(args.cnn_num_sample):
    #             ft = feature_extractor(frames[:, i, :, :, :])
    #             fts.append(ft) 
            
    #         fts = torch.cat(fts, dim=1)
    #         visual_ft.append(fts)
    #         lbl_list.append(labels)
    #     visual_ft = torch.cat(visual_ft, dim=0).cpu().numpy()
    #     lbl_list = torch.cat(lbl_list, dim=0).squeeze()

    # tsne = TSNE(n_jobs=4)
    # result = tsne.fit_transform(visual_ft)
    # plt.figure(figsize=(15,15))
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'black'
    # for i in range(result.shape[0]):
    #     plt.scatter(result[i ,0], result[i, 1], c=colors[lbl_list[i]] )
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('tsne_p1.png')
    
    with open(output_label_dir_p1, 'w+') as p1_valid_txt:
        with torch.no_grad():
            for i, frames in enumerate(test_loader_for_p1):
                frames = frames.cuda()
                fts = []

                for i in range(args.cnn_num_sample):
                    ft = feature_extractor(frames[:, i, :, :, :])
                    fts.append(ft)
                
                fts = torch.cat(fts, dim=1)
                pred = clf(fts.detach().cuda())

                _, pred = torch.max(pred, dim = 1)
                pred = pred.cpu().numpy()

                for i in range(pred.shape[0]):
                    p1_valid_txt.write(str(pred[i]) + '\n')
        


elif args.for_problem == '2':
    p2_clf_resume = 'best_rnn_classifier.pth.tar'
    rnn_clf = RNNClassifier(args).cuda()
    rnn_clf.load_state_dict(torch.load(p2_clf_resume))

    feature_extractor.eval()
    rnn_clf.eval()

    if 'gt_valid.csv' in args.testing_csv_dir[-15:]:
    
        test_loader_for_p2 = torch.utils.data.DataLoader(data.TrimmedVideoData(args, mode='test', model_type='rnn'),
                                                    batch_size=args.train_batch_p2, 
                                                    num_workers=args.workers,
                                                    shuffle=False)
    elif 'gt_test.csv' in args.testing_csv_dir[-15:]:
        test_loader_for_p2 = torch.utils.data.DataLoader(data.TrimmedVideoDataForTesting(args, model_type='rnn'),
                                                    batch_size=args.train_batch_p2, 
                                                    num_workers=args.workers,
                                                    shuffle=False)
        
    output_label_dir_p2 = os.path.join(args.output_label_dir, 'p2_result.txt')

    with open(output_label_dir_p2, 'w+') as p2_result_txt:
        with torch.no_grad():
            fts_list = []
            # visual_ft = [] #
            # gts_list = [] #
            for idx, frames in enumerate(test_loader_for_p2):
                frames = frames.cuda()
                fts = []
                
                for i in range(frames.shape[1]):
                    ft = feature_extractor(frames[:, i, :, :, :])
                    fts.append(ft)
                fts = torch.cat(fts, dim=0)
                fts_list.append(fts)

                if (idx+1) % args.clf_batch == 0:
                    fts_length = [ft.shape[0] for ft in fts_list] 
                    index = np.argsort(-np.array(fts_length))
                    
                    fts_list.sort(key=lambda x: x.shape[0], reverse=True)
                    fts_length = [ft.shape[0] for ft in fts_list]

                    padded_fts = rnn_utils.pad_sequence(fts_list, batch_first=True, padding_value=0) # (batch_size, T, 2048)
                    packed_ft = rnn_utils.pack_padded_sequence(padded_fts, fts_length, batch_first=True).cuda()
                    pred = rnn_clf(packed_ft) # (16, 11)
                    # pred, rnn_ft = rnn_clf(packed_ft) # (16, 11) (8,256)
                    # visual_ft.append(rnn_ft) #
                    _, pred = torch.max(pred, dim = 1) 
                    
                    pred = pred.cpu().numpy()
                    new_pred = np.zeros((args.clf_batch, ), dtype=np.int)
                    for i in range(args.clf_batch):
                        new_pred[index[i]] = pred[i]

                    # gts_list.append(new_pred) #

                    fts_list.clear()

                    for k in range(new_pred.shape[0]):
                        p2_result_txt.write(str(new_pred[k]) + '\n')

                if (idx+1) == len(test_loader_for_p2) and len(test_loader_for_p2) % args.clf_batch != 0:
                    fts_length = [ft.shape[0] for ft in fts_list]
                    index = np.argsort(-np.array(fts_length)) 
                    fts_list.sort(key=lambda x: x.shape[0], reverse=True)
                    fts_length = [ft.shape[0] for ft in fts_list]

                    padded_fts = rnn_utils.pad_sequence(fts_list, batch_first=True, padding_value=0) # (batch_size, T, 2048)
                    packed_ft = rnn_utils.pack_padded_sequence(padded_fts, fts_length, batch_first=True).cuda()
                    pred = rnn_clf(packed_ft) # (16, 11)
                    # pred, rnn_ft = rnn_clf(packed_ft) # (16, 11)
                    # visual_ft.append(rnn_ft) #

                    _, pred = torch.max(pred, dim = 1)
                    
                    pred = pred.cpu().numpy()
                    new_pred = np.zeros((len(index), ), dtype=np.int)
                    for i in range(len(index)):
                        new_pred[index[i]] = pred[i]


                    # gts_list.append(new_pred) #

                    fts_list.clear()

                    for k in range(new_pred.shape[0]):
                        p2_result_txt.write(str(new_pred[k]) + '\n')

            # visual_ft = torch.cat(visual_ft, dim=0).cpu().numpy() #
            # print("v: ", visual_ft.shape)
            # gts_list = np.concatenate(gts_list, axis=0) #
            # print("g: ", gts_list.shape)

            # tsne = TSNE()
            # result = tsne.fit_transform(visual_ft)
            # print("r: ", result.shape)
            # plt.figure(figsize=(15,15))
            # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'black'
            # for i in range(result.shape[0]):
            #     plt.scatter(result[i ,0], result[i, 1], c=colors[gts_list[i]] )
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig('tsne_p2.png')

elif args.for_problem == '3':
    p3_clf_resume = 'best_rnn_classifier3.pth.tar'
    rnn_clf = RNNClassifierForP3(args).cuda()
    rnn_clf.load_state_dict(torch.load(p3_clf_resume))

    feature_extractor.eval()
    rnn_clf.eval()

    test_loader_for_p3 = torch.utils.data.DataLoader(data.FullLengthVideoDataForTesting(args),
                                                    batch_size=args.train_batch_p2, 
                                                    num_workers=args.workers,
                                                    shuffle=False)

    with torch.no_grad():
        for frames, categ_name in test_loader_for_p3:
            categ_name = categ_name[0]
            categ_pred_txt_file = os.path.join(args.output_label_dir, categ_name) + '.txt'
            with open(categ_pred_txt_file, 'w+') as pred_txt_file:
                frames = frames.cuda()
                fts = []
                
                for i in range(frames.shape[1]):
                    ft = feature_extractor(frames[:, i, :, :, :])
                    fts.append(ft)
                
                fts = torch.cat(fts, dim=0).unsqueeze(dim=0)
                pred = rnn_clf(fts) # (16, 11)

                _, pred = torch.max(pred, dim = 1)
                pred = pred.cpu().numpy()

                for k in range(pred.shape[0]):
                    pred_txt_file.write(str(pred[k]) + '\n')



