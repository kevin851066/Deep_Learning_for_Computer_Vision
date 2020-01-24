# TODO: create shell script for Problem 2

wget https://www.dropbox.com/s/iz95zt6pxt4g2hs/best_rnn_classifier.pth.tar?dl=1 -O best_rnn_classifier.pth.tar
python3 testing4all.py --testing_data_dir $1 --testing_csv_dir $2 --output_label_dir $3 --for_problem '2'
