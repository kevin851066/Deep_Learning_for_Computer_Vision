import os
import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter

import lib.parser as parser
import lib.baseline_models as model
import lib.baseline_data as data
from lib.baseline_trainer import Trainer

if __name__ == '__main__':

    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_cuda = torch.cuda.is_available()

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
                                                   shuffle=True)
    val_gallery_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid', type='gallery'),
                                                     batch_size=args.batch_size,
                                                     num_workers=args.workers,
                                                     shuffle=True)

    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
    print('===> Preparing model...')
    my_model = model.Model()
    if use_cuda:
        my_model.cuda()

    optim = torch.optim.Adam([
        {'params': my_model.parameters()},
    ], lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

    my_trainer = Trainer(model=my_model,
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
