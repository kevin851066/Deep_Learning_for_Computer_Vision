#!/bin/bash
python3 PGSA_train.py -e 50 -b 64 --lr 2e-4 --metric 'L2' --tri_loss --alpha 0.5 \
		      --train_csv '../final/data/reid_mask_list_train.csv' \
		      --query_csv '../final/data/query.csv' --gallery_csv '../final/data/gallery.csv'\
		      --mask_dir '../final/data/mask_2' \
	              --train_img_dir '../final/data/train' --eval_img_dir '../final/data/imgs'\
		      --save_dir 'ckpts/PGSA_tri_3_alpha_0.5_conv'
