import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

import lib.parser as parser
import lib.PGSA_models as model
import lib.PGSA_data as data
from lib.PGSA_trainer import Trainer

if __name__ == '__main__':

    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    print('===> Preparing data...')

    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_query_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid', type='query'),
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=False)
    val_gallery_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid', type='gallery'),
                                                     batch_size=args.batch_size,
                                                     num_workers=args.workers,
                                                     shuffle=False)

    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
    print('===> Preparing model...')
    my_models = {
        'seg_model': model.Seg_Model(),
    }
    my_models['reid_model'] = model.ReID_Model(seg_extractor=my_models['seg_model'].extractor if args.share_E else None)
    if use_cuda:
        for _, m in my_models.items():
            m.cuda()

    optim = torch.optim.Adam([
        {'params': my_models['seg_model'].parameters()},
        {'params': my_models['reid_model'].parameters()},
    ], lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

    # optim = torch.optim.Adam(list(my_models['seg_model'].parameters()) + list(my_models['reid_model'].parameters()),
    #                          lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

    my_trainer = Trainer(models=my_models,
                         train_loader=train_loader,
                         val_query_loader=val_query_loader,
                         val_gallery_loader=val_gallery_loader,
                         optim=optim,
                         args=args,
                         writer=writer,
                         use_cuda=use_cuda
                         )

    print("===> Initializing training...")
    my_trainer.train()
