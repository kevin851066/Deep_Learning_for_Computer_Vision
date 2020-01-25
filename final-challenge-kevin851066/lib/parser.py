from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='My parser for Re-ID')

    # Datasets parameters
    parser.add_argument('--img_dir', type=str, default='data',
                        help="root path to img directory")
    parser.add_argument('--train_csv', type=str, default='train.csv',
                        help="path to train.csv")
    parser.add_argument('--query_csv', type=str, default='query.csv',
                        help="path to query.csv")
    parser.add_argument('--gallery_csv', type=str, default='gallery.csv',
                        help="path to gallery.csv")
    parser.add_argument('--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--train_img_dir', type=str, default='data',
                        help="path to training img directory")
    parser.add_argument('--eval_img_dir', type=str, default='data',
                        help="path to evaluating img directory")
    parser.add_argument('--mask_dir', type=str, default='data/mask',
                        help="path to mask directory (for PGSA)")

    # training parameters
    parser.add_argument('--epochs', '-e', default=100, type=int,
                        help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                        help="num of validation iterations")
    parser.add_argument('--save_epoch', default=10, type=int,
                        help="num of save iterations")
    parser.add_argument('--batch_size', '-b', default=32, type=int,
                        help="batch size")
    parser.add_argument('--lr', '-lr', default=0.0002, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight_decay', default=0.0005, type=float,
                        help="rate of weight decay")
    parser.add_argument('--metric', default='L2', choices=['L2', 'cosine'], type=str,
                        help='the metric to measure distance')
    parser.add_argument('--tri_loss', action='store_true',
                        help='whether to use triplet loss or not')
    parser.add_argument('--id_loss', action='store_true',
                        help='whether to use id loss or not')
    parser.add_argument('--share_E', action='store_true',
                        help='whether to share extractors or not')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help="scale factor for tri_loss")

    # resume trained model
    parser.add_argument('--pretrained', type=str, default='',
                        help="path to the trained model")
    parser.add_argument('--resume', action='store_true',
                        help="resume training")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    # model
    parser.add_argument('--n_classes', default=10, type=int,
                        help="# of classes")

    # infer
    parser.add_argument('--infer_data_dir', type=str, default='')
    parser.add_argument('--csv_path', type=str, default='')

    # for tSNE
    parser.add_argument('--tSNE_model', type=str, default='DANN', choices=['DANN', 'ADDA'])

    args = parser.parse_args()

    return args
