# ADDA

from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='ADDA')

    # Datasets parameters
    
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    # others
    parser.add_argument('--testing_img_dir', type=str)
    parser.add_argument('--testing_type', type=str)
    parser.add_argument('--pred_csv_dir', type=str)

    args = parser.parse_args()

    return args