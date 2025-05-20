#!/bin/bash

tmux new-session -d -s test
tmux send-keys "conda activate linkdom" C-m
tmux send-keys "CUDA_VISIBLE_DEVICES=0,1 python test.py model=vqvae exp_model_name=exp-vqvae-imagenet-0/2024-12-07-22-50-03" C-m

tmux new-window -t test
tmux send-keys "conda activate linkdom" C-m
tmux send-keys "CUDA_VISIBLE_DEVICES=2,3 python test.py model=amvqvae exp_model_name=exp-amvqvae-1/2024-12-14-22-51-13" C-m

tmux new-window -t test
tmux send-keys "conda activate linkdom" C-m
tmux send-keys "CUDA_VISIBLE_DEVICES=4,5 python test.py model=smvqvae exp_model_name=exp-smvqvae-1/2024-12-14-22-52-48" C-m

tmux new-window -t test
tmux send-keys "conda activate linkdom" C-m
tmux send-keys "CUDA_VISIBLE_DEVICES=6,7 python test.py model=vqvae exp_model_name=exp-vqvae-imagenet-2/2024-12-13-23-06-25" C-m

tmux select-window -t test:0
tmux attach-session -d