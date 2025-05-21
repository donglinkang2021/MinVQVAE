tmux new-session -s "linkdom-vqvae"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=0,1
python train.py task.model.quantize._target_="minvqvae.core.SoftmaxQuantize"
python train.py task.model.quantize._target_="minvqvae.core.FFN"
python train.py task.model.quantize._target_="minvqvae.core.AttentionQuantize"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=2,3
python train.py task.model.quantize._target_="minvqvae.core.ArgmaxQuantize"
python train.py task.model.quantize._target_="minvqvae.core.LLMFFN"
python train.py \
    task.model.quantize._target_="minvqvae.core.SoftmaxQuantize" \
    task.model.scale_factor=2

conda activate linkdom
export CUDA_VISIBLE_DEVICES=4,5
python train.py task.model.quantize._target_="minvqvae.core.SoftQuantize"
python train.py task.model.quantize._target_="minvqvae.core.Identity"
python train.py \
    task.model.quantize._target_="minvqvae.core.Identity" \
    task.model.scale_factor=2

conda activate linkdom
export CUDA_VISIBLE_DEVICES=6,7
python train.py task.model.quantize._target_="minvqvae.core.SimpleQuantize"
python train.py \
    task.model.quantize._target_="minvqvae.core.AttentionQuantize" \
    task.model.scale_factor=2
