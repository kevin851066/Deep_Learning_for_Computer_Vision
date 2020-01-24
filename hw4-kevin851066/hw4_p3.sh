# TODO: create shell script for Problem 3

wget https://www.dropbox.com/s/kdz9464xdh9yn7f/best_rnn_classifier3.pth.tar?dl=1 -O best_rnn_classifier3.pth.tar
python3 testing4all.py --testing_data_dir $1 --output_label_dir $2 --for_problem '3'
