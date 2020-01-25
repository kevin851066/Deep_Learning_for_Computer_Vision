import torch
import os
import numpy as np
import lib.baseline_data as data
import lib.baseline_models as model
import lib.parser as parser
import lib.utils as utils
import pandas as pd

def predict(args, model, use_cuda, test_gallery_loader, test_query_loader):
    if use_cuda:
        model.cuda()
    model.eval()
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        '''gather gallery features'''
        gallery_feats = []
        gallery_names = []
        for idx, (imgs, names) in enumerate(test_gallery_loader):
            if use_cuda:
                imgs = imgs.cuda()
            feat, _ = model(imgs)
            gallery_feats.append(feat)
            gallery_names.append(np.asarray(names))

        gallery_feats = torch.cat(gallery_feats, dim=0)
        gallery_names = np.concatenate(gallery_names, axis=0)
        '''gather features for each batch of queries'''
        query_feats = []
        for idx, (imgs, names) in enumerate(test_query_loader):
            if use_cuda:
                imgs = imgs.cuda()
            feat, _ = model(imgs)
            query_feats.append(feat)

        query_feats = torch.cat(query_feats, dim=0)
    dist = utils.get_pairwise_distance(query_feats, gallery_feats, metric=args.metric)
    top_idx = dist.argmin(dim=1).cpu().numpy()
    print(top_idx)
    pred = gallery_names[top_idx]
    save_csv(args, preds=pred)

def save_csv(args, preds):
    print(preds)
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
    m = model.Model()
    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        print("===>Loading pretrained model {}...".format(args.pretrained))
        m.load_state_dict(checkpoint['model'])
        if args.resume:
            iters = checkpoint['iters']
        #checkpoint = torch.load(args.pretrained)
        #m.load_state_dict(checkpoint)
    predict(args, model=m, use_cuda=use_cuda, test_gallery_loader=gallery_loader, test_query_loader=query_loader)
