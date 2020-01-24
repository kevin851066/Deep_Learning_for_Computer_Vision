# ACGAN

from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='ACGAN')

    # Datasets parameters
    
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_img_dir', type=str)

    args = parser.parse_args()

    return args