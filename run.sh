# download data
nohup python download_dataset.py > download.log 2>&1 &

# train model
nohup python train.py > train.log 2>&1 &

# read log
python read_nohup.py --file train.log

# tensorboard
nohup tensorboard --logdir=./logs --bind_all > tensorboard.log 2>&1 &
