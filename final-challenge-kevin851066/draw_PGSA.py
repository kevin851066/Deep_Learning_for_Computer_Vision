import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import lib.PGSA_data as data
import lib.PGSA_models as model
import lib.parser as parser
from torchvision.utils import save_image
from lib.utils import denormalize, normalize

label_map = {
    (0, 255, 255): 1,
    (255, 0, 0): 2,
    (0, 0, 255): 3,
    (255, 255, 255): 0
}

rev_label_map = {
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 0, 255),
    0: (255, 255, 255)
}


def draw_mask(seg_map, save_name=''):
    print('Drawing {}...'.format(save_name))
    seg_map = seg_map.cpu().numpy()
    canvas = np.zeros((*seg_map.shape, 3))
    for lbl, _ in rev_label_map.items():
        canvas[seg_map == lbl] = rev_label_map[lbl]
    canvas = canvas.squeeze()
    print(canvas.shape)
    if len(save_name):
        cv2.imwrite(save_name, canvas)


def predict(args, use_cuda, train_loader, model_reid=None, model_seg=None):
    if use_cuda:
        model_reid.cuda()
        model_seg.cuda()
    model_reid.eval()
    model_seg.eval()
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        '''gather gallery features'''
        for idx, (imgs, gts, lbls) in enumerate(train_loader):
            if idx > 0:
                break
            if use_cuda:
                imgs, gts, lbls = imgs.cuda(), gts.cuda(), lbls.cuda()
            global_feat, seg_maps = model_seg(imgs)
            _, pred = torch.max(seg_maps, dim=1)
            feat, _ = model_reid(imgs, seg_maps, global_feat, denorm=True)

            # draw mask
            draw_mask(seg_map=gts, save_name=os.path.join(args.save_dir, 'mask_gt.jpg'))
            draw_mask(seg_map=pred, save_name=os.path.join(args.save_dir, 'mask_seg.jpg'))

            # draw attention
            softmax = nn.Softmax2d().cuda()
            seg_maps = softmax(seg_maps)
            imgs_den = denormalize(imgs)
            save_image(imgs_den, os.path.join(args.save_dir, 'image.jpg'),
                       nrow=8, normalize=True)
            imgs_att = imgs_den * seg_maps[:, 1, :, :].unsqueeze(1)
            save_image(imgs_att, os.path.join(args.save_dir, 'att_trunk.jpg'),
                       nrow=8, normalize=True)

            imgs_att = imgs_den * seg_maps[:, 2, :, :].unsqueeze(1)
            save_image(imgs_att, os.path.join(args.save_dir, 'att_f_leg.jpg'),
                       nrow=8, normalize=True)

            imgs_att = imgs_den * seg_maps[:, 3, :, :].unsqueeze(1)
            save_image(imgs_att, os.path.join(args.save_dir, 'att_h_leg.jpg'),
                       nrow=8, normalize=True)


if __name__ == "__main__":
    args = parser.arg_parse()
    use_cuda = torch.cuda.is_available()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    print('===> Preparing data...')
    # TODO: change data mode to test
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                                     batch_size=1,
                                                     num_workers=args.workers,
                                                     shuffle=True)
    print('===> Preparing model...')
    seg = model.Seg_Model()
    reid = model.ReID_Model()
    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        print("===>Loading pretrained model {}...".format(args.pretrained))
        seg.load_state_dict(checkpoint['seg_model'])
        reid.load_state_dict(checkpoint['reid_model'])
    predict(args, model_reid=reid, model_seg=seg, use_cuda=use_cuda, train_loader=train_loader)
