import torch
import os
import numpy as np
import lib.PGSA_data as data
import lib.PGSA_models as model
import lib.parser as parser
import lib.utils as utils
import pandas as pd

def predict(args, use_cuda, test_gallery_loader, test_query_loader, model_reid=None, model_seg=None):
    if use_cuda:
        model_reid.cuda()
        model_seg.cuda()
    model_reid.eval()
    model_seg.eval()
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        '''gather gallery features'''
        gallery_feats = []
        gallery_names = []
        for idx, (imgs, names) in enumerate(test_gallery_loader):
            if use_cuda:
                imgs = imgs.cuda()
            global_feat, seg_maps = model_seg(imgs)
            feat, _ = model_reid(imgs, seg_maps, global_feat, denorm=True)
            gallery_feats.append(feat)
            gallery_names.append(np.asarray(names))

        gallery_feats = torch.cat(gallery_feats, dim=0)
        gallery_names = np.concatenate(gallery_names, axis=0)
        '''gather features for each batch of queries'''
        query_feats = []
        for idx, (imgs, names) in enumerate(test_query_loader):
            if use_cuda:
                imgs = imgs.cuda()
            global_feat, seg_maps = model_seg(imgs)
            feat, _ = model_reid(imgs, seg_maps, global_feat, denorm=True)
            query_feats.append(feat)

        query_feats = torch.cat(query_feats, dim=0)
    dist = utils.get_pairwise_distance(query_feats, gallery_feats, metric=args.metric)
    top_idx = dist.argmin(dim=1).cpu().numpy()
    pred = gallery_names[top_idx]
    save_csv(args, preds=pred)


def save_csv(args, preds):
    pd.DataFrame(preds).to_csv(args.csv_path, index=False, header=False)


if __name__ == "__main__":
    args = parser.arg_parse()
    use_cuda = torch.cuda.is_available()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    print('===> Preparing data...')
    # TODO: change data mode to test
    query_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test', type='query'),
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=False)
    gallery_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test', type='gallery'),
                                                     batch_size=args.batch_size,
                                                     num_workers=args.workers,
                                                     shuffle=False)
    print('===> Preparing model...')
    seg = model.Seg_Model()
    reid = model.ReID_Model()
    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        print("===>Loading pretrained model {}...".format(args.pretrained))
        seg.load_state_dict(checkpoint['seg_model'])
        reid.load_state_dict(checkpoint['reid_model'])
    predict(args, model_reid=reid, model_seg=seg, use_cuda=use_cuda, test_gallery_loader=gallery_loader, test_query_loader=query_loader)
