wget https://www.dropbox.com/s/khyil0pu5v4urzh/m_s_dann_best.pth.tar?dl=1 -O m_s_dann_best.pth.tar
wget https://www.dropbox.com/s/h3twymjs0idq6dv/s_m_dann_best.pth.tar?dl=1 -O s_m_dann_best.pth.tar
python3 dann_testing.py --testing_img_dir $1 --testing_type $2 --pred_csv_dir $3
