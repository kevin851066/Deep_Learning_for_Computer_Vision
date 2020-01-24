# TODO: create shell script for running the testing code of your improved model

wget https://www.dropbox.com/s/id2gwlpzafeqf22/model_best.pth.tar?dl=1 -O strong_baseline_model_best.pth.tar

# https://drive.google.com/file/d/17O8gaItBT38axzJcpUlsqZvlgjHAikTT/view?usp=sharing
RESUME='strong_baseline_model_best.pth.tar'
python3 improved_test.py --resume $RESUME --data_dir $1 --save_dir $2