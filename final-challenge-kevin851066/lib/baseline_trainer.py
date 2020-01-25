import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from .utils import HardTripletLoss, evaluate_rank1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Trainer:
    def __init__(self, model, train_loader, val_query_loader, val_gallery_loader, optim, args, writer, use_cuda):
        self.model = model
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
            self.model.load_state_dict(checkpoint['model'])
            if self.args.resume:
                self.iters = checkpoint['iters']

    def get_lr(self):
        return self.base_lr * (1 - self.iters / self.max_iter) ** 0.9

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.total_cnt, self.correct_cnt = 0, 0
            current_iter = 0
            print('')

            '''set new lr'''
            for param in self.optim.param_groups:
                print('Epoch {}, New lr: {}'.format(epoch, self.get_lr()))
                param['lr'] = self.get_lr()

            '''train an epoch'''
            for idx, (imgs, lbls) in enumerate(self.train_loader):
                self.iters += 1
                current_iter += 1
                if self.use_cuda:
                    imgs, lbls = imgs.cuda(), lbls.cuda()  # video: (B, T, C, H, W)
                lbls = lbls.squeeze()

                '''model forwarding & loss calculation'''
                feat, cls_out = self.model(imgs)
                id_loss = self.id_criterion(cls_out, lbls)
                tri_loss = self.tri_criterion(feat, lbls)

                loss = []
                if self.args.id_loss:
                    loss.append(id_loss)
                if self.args.tri_loss:
                    loss.append(tri_loss)
                assert len(loss) > 0
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
                    print('Epoch [{}][{}/{}], id_loss: {:.4f}, tri_loss: {:.4f}'.format(epoch, current_iter,
                                                                                        len(self.train_loader),
                                                                                        id_loss.item(),
                                                                                        tri_loss.item()))
                torch.cuda.empty_cache()

            train_acc = self.correct_cnt / self.total_cnt
            self.writer.add_scalar('acc/train_id_acc', train_acc, epoch)
            print('Epoch {}, Train ID Acc: {:.4f}'.format(epoch, train_acc))

            if epoch % self.val_epoch == 0:
                self.evaluate(epoch)

            if epoch % self.save_epoch == 0:
                torch.save({
                    'model': self.model.state_dict(),
                    'iters': self.iters
                }, os.path.join(self.save_dir, 'checkpoint_{}.pth.tar'.format(epoch)))

        print('\nBest Val Rank-1 Acc: {:.5f}'.format(self.best_acc))

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():  # do not need to calculate information for gradient during eval
            '''gather gallery features'''
            gallery_feats = []
            gallery_ids = []
            for idx, (imgs, gts) in enumerate(self.val_gallery_loader):
                if self.use_cuda:
                    imgs, gts = imgs.cuda(), gts.cuda()
                feat, _ = self.model(imgs)
                gallery_feats.append(feat)
                gallery_ids.append(gts)

            gallery_feats = torch.cat(gallery_feats, dim=0)
            gallery_ids = torch.cat(gallery_ids, dim=0).view(-1)

            '''gather features for each batch of queries'''
            query_feats = []
            query_ids = []
            for idx, (imgs, gts) in enumerate(self.val_query_loader):
                if self.use_cuda:
                    imgs, gts = imgs.cuda(), gts.cuda()
                feat, _ = self.model(imgs)
                query_feats.append(feat)
                query_ids.append(gts)

            query_feats = torch.cat(query_feats, dim=0)
            query_ids = torch.cat(query_ids, dim=0).view(-1)

        val_rank1_acc = evaluate_rank1(query_feats, query_ids, gallery_feats, gallery_ids, metric=self.metric)
        self.writer.add_scalar('acc/val_rank1_acc', val_rank1_acc, epoch)
        print('Epoch {}, Val Rank-1 Acc: {:.4f}'.format(epoch, val_rank1_acc))

        if val_rank1_acc > self.best_acc:
            torch.save({
                'model': self.model.state_dict(),
                'iters': 0
            }, os.path.join(self.save_dir, 'checkpoint_best.pth.tar'))
            self.best_acc = val_rank1_acc
