# Hw4
from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='For task 1 of hw4')

    # Datasets parameters
    parser.add_argument('--trimmed_video_data_dir', type=str, default='../hw4_data/TrimmedVideos/', 
                    help="root path to trimmed video data directory")
    parser.add_argument('--full_length_video_data_dir', type=str, default='../hw4_data/FullLengthVideos/', 
                    help="root path to full length video data directory")
        
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    # parser.add_argument('--epoch', default=100, type=int,
    #                 help="num of validation iterations")
    # parser.add_argument('--val_epoch', default=1, type=int,
    #                 help="num of validation iterations")
    parser.add_argument('--train_batch', default=16, type=int,
                    help="train batch size")
    parser.add_argument('--train_batch_p2', default=1, type=int,
                    help="train batch size")      
    parser.add_argument('--clf_batch', default=8, type=int,
                    help="classifier batch size")
    # parser.add_argument('--lr', default=0.0002, type=float,
    #                 help="initial learning rate")
    # parser.add_argument('--beta1', default=0.5, type=float,
    #                 help="parameter1 for adam")
    # parser.add_argument('--beta2', default=0.999, type=float,
    #                 help="parameter2 for adam")
    
    # gru parameters
    parser.add_argument('--hidden_size', default=256, type=int,
                    help="hidden size of GRU")
    parser.add_argument('--num_layer', default=2, type=int,
                    help="number of layers of GRU")
    parser.add_argument('--num_layer_p3', default=1, type=int,
                    help="number of layers of GRU")
    
    # data loader
    parser.add_argument('--cnn_num_sample', default=2, type=int,
                    help="number of frame sample in cnn")
    parser.add_argument('--rnn_num_sample', default=10, type=int,
                    help="number of frame sample in rnn")
    # for p3 data loader
    parser.add_argument('--rnn_num_frame', default=256, type=int,
                    help="number of frame sample of each video")
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    
    # testing
    parser.add_argument('--testing_data_dir', type=str)
    parser.add_argument('--testing_csv_dir', type=str)  
    parser.add_argument('--output_label_dir', type=str) 
    parser.add_argument('--for_problem', type=str) 
    
    # others
    parser.add_argument('--save_dir', type=str, default='tmp')
    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args
