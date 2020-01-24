# TODO: create shell script for running the testing code of the baseline model

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=16OtY_Ju4zZS--zzVM6hMZChHMplGvHC8' -O baseline_model_best.pth.tar

RESUME='baseline_model_best.pth.tar'
python3 test.py --resume $RESUME --data_dir $1 --save_dir $2