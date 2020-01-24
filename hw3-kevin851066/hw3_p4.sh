wget https://www.dropbox.com/s/kx03dvw017zd4z2/adda_ms_src_clf.pth.tar?dl=1 -O adda_ms_src_clf.pth.tar
wget https://www.dropbox.com/s/etek92s4bxsq7ur/adda_ms_tgt_enc.pth.tar?dl=1 -O adda_ms_tgt_enc.pth.tar
wget https://www.dropbox.com/s/dognx1q7ucpr7rt/adda_sm_src_clf.pth.tar?dl=1 -O adda_sm_src_clf.pth.tar
wget https://www.dropbox.com/s/s60xiropw4h4rd1/adda_sm_tgt_enc.pth.tar?dl=1 -O adda_sm_tgt_enc.pth.tar
python3 adda_testing.py --testing_img_dir $1 --testing_type $2 --pred_csv_dir $3
