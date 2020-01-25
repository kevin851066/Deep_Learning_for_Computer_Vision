#!/bin/bash
python3 baseline_train.py -e 100 -b 64 --lr 2e-5 --metric 'cosine'\
			  --train_csv 'data/train.csv' --query_csv 'data/query.csv' --gallery_csv 'data/gallery.csv'\
	                  --img_dir 'data/imgs' --save_dir 'ckpts/base_cosine/'
