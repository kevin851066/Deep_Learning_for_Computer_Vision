WGET='https://www.dropbox.com/s/7igvlyh6daaa5vy/checkpoint_best.pth.tar?dl=1'
DIR='checkpoint_best.pth.tar'
wget -O $DIR $WGET 
python3 infer_PGSA.py --resume --pretrained checkpoint_best.pth.tar --eval_img_dir $1 --train_csv " " --query_csv $2 --gallery_csv $3 --csv_path $4 --metric 'L2'






