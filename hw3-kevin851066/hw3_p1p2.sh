wget https://www.dropbox.com/s/shflu56ai2vl65u/DCGAN.pth.tar?dl=1 -O DCGAN.pth.tar

python3 dcgan_testing.py --save_img_dir $1

wget https://www.dropbox.com/s/hdachu2vqijysmw/ACGAN.pth.tar?dl=1 -O ACGAN.pth.tar

python3 acgan_testing.py --save_img_dir $1
