import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from .utils import HardTripletLoss, evaluate_rank1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


rev_label_map = {
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (0, 255, 0),
    0: (255, 255, 255)
}


class Trainer:
    def __init__(self, models, train_loader, val_query_loader, val_gallery_loader, optim, args, writer, use_cuda):
        assert isinstance(models, dict)
        self.seg_model = models['seg_model']
        self.reid_model = models['reid_model']

        # self.seg_model = nn.DataParallel(self.seg_model)
        # self.reid_model = nn.DataParallel(self.reid_model)

        self.train_loader = train_loader
        self.val_query_loader = val_query_loader
        self.val_gallery_loader = val_gallery_loader
        self.optim = optim
        self.args = args
        self.writer = writer
        self.use_cuda = use_cuda

        self.epochs = self.args.epochs
        self.val_epoch = self.args.val_epoch
        self.save_epoch = self.args.save_epoch
        self.save_dir = self.args.save_dir
        self.base_lr = self.args.lr
        self.metric = self.args.metric

        self.seg_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 5, 5]).float().cuda())
        self.id_criterion = nn.CrossEntropyLoss()
        self.tri_criterion = HardTripletLoss(metric=self.metric)

        self.iters = 0
        self.max_iter = len(self.train_loader) * self.epochs
        self.best_acc = 0
        self.total_cnt = 0
        self.correct_cnt = 0

        if self.args.pretrained:
            checkpoint = torch.load(self.args.pretrained)
            print("===>Loading pretrained model {}...".format(self.args.pretrained))
            self.seg_model.load_state_dict(checkpoint['seg_model'])
            self.reid_model.load_state_dict(checkpoint['reid_model'])
            if self.args.resume:
                self.iters = checkpoint['iters']

    def get_lr(self):
        return self.base_lr * (1 - self.iters / self.max_iter) ** 0.9

    def get_alpha(self):
        return self.args.alpha
        # return min(3 * self.iters / self.max_iter, 1)

    def train(self):
        self.evaluate(0)
        for epoch in range(1, self.epochs + 1):
            self.seg_model.train()
            self.reid_model.train()
            self.total_cnt, self.correct_cnt = 0, 0
            current_iter = 0
            print('')

            '''set new lr'''
            print('Epoch {}, New lr: {}'.format(epoch, self.get_lr()))
            for param in self.optim.param_groups:
                param['lr'] = self.get_lr()

            '''train an epoch'''
            for idx, (imgs, img_seg_maps, lbls) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1
                if self.use_cuda:
                    imgs, img_seg_maps, lbls = imgs.cuda(), img_seg_maps.cuda(), lbls.cuda()  # imgs: (B, C, H, W)
                lbls = lbls.squeeze()

                '''model forwarding & loss calculation'''
                global_feat, seg_out = self.seg_model(imgs)  # seg_out: (B, seg_n_classes, H, W)
                seg_loss = self.seg_criterion(seg_out, img_seg_maps)

                feat, cls_out = self.reid_model(imgs, seg_out, global_feat)
                id_loss = self.id_criterion(cls_out, lbls)
                tri_loss = self.tri_criterion(feat, lbls)

                loss = [seg_loss]
                if self.args.id_loss:
                    loss.append(id_loss)
                if self.args.tri_loss:
                    loss.append(tri_loss * self.get_alpha())
                loss = sum(loss)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    _, preds = torch.max(cls_out, dim=1)
                    self.total_cnt += preds.cpu().numpy().size
                    self.correct_cnt += (preds == lbls).sum().item()

                self.writer.add_scalar('loss', loss.item(), self.iters)
                if current_iter % 5 == 0 or current_iter == len(self.train_loader):
                    print('Epoch [{}][{}/{}], seg_loss: {:.4f}, id_loss: {:.4f}, tri_loss: {:.4f}'.format(
                        epoch, current_iter, len(self.train_loader), seg_loss.item(), id_loss.item(), tri_loss.item()))
                torch.cuda.empty_cache()

            train_acc = self.correct_cnt / self.total_cnt
            self.writer.add_scalar('acc/train_id_acc', train_acc, epoch)
            print('Epoch {}, Train ID Acc: {:.4f}'.format(epoch, train_acc))

            if epoch % self.val_epoch == 0:
                self.evaluate(epoch)

            if epoch % self.save_epoch == 0:
                torch.save({
                    'seg_model': self.seg_model.state_dict(),
                    'reid_model': self.reid_model.state_dict(),
                    'iters': self.iters
                }, os.path.join(self.save_dir, 'checkpoint_{}.pth.tar'.format(epoch)))

        print('\nBest Val Rank-1 Acc: {:.5f}'.format(self.best_acc))

    def evaluate(self, epoch):
        self.seg_model.eval()
        self.reid_model.eval()
        with torch.no_grad():  # do not need to calculate information for gradient during eval
            '''gather gallery features'''
            gallery_feats = []
            gallery_ids = []
            total_cnt = 0
            correct_cnt = 0
            for idx, (imgs, img_masks, gts, img_name) in enumerate(self.val_gallery_loader):
                if self.use_cuda:
                    imgs, img_masks, gts = imgs.cuda(), img_masks.cuda(), gts.cuda()
                global_feat, seg_out = self.seg_model(imgs)  # (B, seg_n_classes, H, W)
                feat, cls_out = self.reid_model(imgs, seg_out, global_feat)

                gallery_feats.append(feat)
                gallery_ids.append(gts)

                # compute seg acc
                max_values, _ = torch.max(img_masks.view(img_masks.shape[0], -1), dim=1)
                valid_idx = max_values > 0

                _, pred = torch.max(seg_out, dim=1)
                total_cnt += pred[valid_idx].cpu().numpy().size
                correct_cnt += (pred[valid_idx] == img_masks[valid_idx]).cpu().sum().item()

                # draw 1 image
                if idx < 1:
                    softmax = nn.Softmax2d().cuda()
                    seg_out = softmax(seg_out)
                    imgs_att = imgs * seg_out[:, 1, :, :].unsqueeze(1)
                    save_image(imgs_att, os.path.join(self.save_dir, '{}_att_trunk.jpg'.format(epoch)),
                               nrow=8, normalize=True)
                    # imgs_att = imgs * seg_out[:, 2, :, :].unsqueeze(1)
                    # save_image(imgs_att, os.path.join(self.save_dir, '{}_att_leg.jpg'.format(epoch)),
                    #            nrow=8, normalize=True)

                    imgs_att = imgs * seg_out[:, 2, :, :].unsqueeze(1)
                    save_image(imgs_att, os.path.join(self.save_dir, '{}_att_f_leg.jpg'.format(epoch)),
                               nrow=8, normalize=True)

                    imgs_att = imgs * seg_out[:, 3, :, :].unsqueeze(1)
                    save_image(imgs_att, os.path.join(self.save_dir, '{}_att_h_leg.jpg'.format(epoch)),
                               nrow=8, normalize=True)

            gallery_feats = torch.cat(gallery_feats, dim=0)
            gallery_ids = torch.cat(gallery_ids, dim=0).view(-1)

            '''gather features for each batch of queries'''
            query_feats = []
            query_ids = []
            for idx, (imgs, img_masks, gts, img_name) in enumerate(self.val_query_loader):
                if self.use_cuda:
                    imgs, img_masks, gts = imgs.cuda(), img_masks.cuda(), gts.cuda()
                global_feat, seg_out = self.seg_model(imgs)  # (B, seg_n_classes, H, W)
                feat, cls_out = self.reid_model(imgs, seg_out, global_feat)

                query_feats.append(feat)
                query_ids.append(gts)

                # compute seg acc
                max_values, _ = torch.max(img_masks.view(img_masks.shape[0], -1), dim=1)
                valid_idx = max_values > 0

                _, pred = torch.max(seg_out, dim=1)
                total_cnt += pred[valid_idx].cpu().numpy().size
                correct_cnt += (pred[valid_idx] == img_masks[valid_idx]).cpu().sum().item()

            query_feats = torch.cat(query_feats, dim=0)
            query_ids = torch.cat(query_ids, dim=0).view(-1)

        val_seg_acc = correct_cnt / total_cnt
        self.writer.add_scalar('acc/val_seg_acc', val_seg_acc, epoch)
        print('Epoch {}, Val Seg Acc: {:.4f}'.format(epoch, val_seg_acc))

        val_rank1_acc = evaluate_rank1(query_feats, query_ids, gallery_feats, gallery_ids, metric=self.metric)
        self.writer.add_scalar('acc/val_rank1_acc', val_rank1_acc, epoch)
        print('Epoch {}, Val Rank-1 Acc: {:.4f}, Best Acc: {:.4f}'.format(epoch, val_rank1_acc, self.best_acc))

        if val_rank1_acc > self.best_acc:
            torch.save({
                'seg_model': self.seg_model.state_dict(),
                'reid_model': self.reid_model.state_dict(),
                'iters': 0
            }, os.path.join(self.save_dir, 'checkpoint_best.pth.tar'))
            self.best_acc = val_rank1_acc
