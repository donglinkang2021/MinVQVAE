tmux new-session -s "linkdom-vqvae"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=0,1
python train.py task.model.quantize._target_="minvqvae.core.SoftmaxQuantize"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=2,3
python train.py task.model.quantize._target_="minvqvae.core.ArgmaxQuantize"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=4,5
python train.py task.model.quantize._target_="minvqvae.core.SoftQuantize"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=6,7
python train.py task.model.quantize._target_="minvqvae.core.SimpleQuantize"

