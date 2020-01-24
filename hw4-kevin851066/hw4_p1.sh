# TODO: create shell script for Problem 1
wget https://www.dropbox.com/s/eyk51tgphlxjel9/best_classifier.pth.tar?dl=1 -O best_classifier.pth.tar
python3 testing4all.py --testing_data_dir $1 --testing_csv_dir $2 --output_label_dir $3 --for_problem '1'