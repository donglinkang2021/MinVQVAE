tmux new-session -s "linkdom-vqvae"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=0,1
python train.py model.quantize._target_="minvqvae.core.SoftmaxQuantize"
python train.py model.quantize._target_="minvqvae.core.FFN"
python train.py model.quantize._target_="minvqvae.core.AttentionQuantize"
python train.py model=vanilla_vae # kl_weight=0.5

conda activate linkdom
export CUDA_VISIBLE_DEVICES=2,3
python train.py model.quantize._target_="minvqvae.core.ArgmaxQuantize"
python train.py model.quantize._target_="minvqvae.core.LLMFFN"
python train.py \
    model.quantize._target_="minvqvae.core.SoftmaxQuantize" \
    model.scale_factor=2
python train.py model=vanilla_vae # kl_weight=1

conda activate linkdom
export CUDA_VISIBLE_DEVICES=4,5
python train.py model.quantize._target_="minvqvae.core.SoftQuantize"
python train.py model.quantize._target_="minvqvae.core.Identity"
python train.py \
    model.quantize._target_="minvqvae.core.Identity" \
    model.scale_factor=2

conda activate linkdom
export CUDA_VISIBLE_DEVICES=6,7
python train.py model.quantize._target_="minvqvae.core.SimpleQuantize"
python train.py \
    model.quantize._target_="minvqvae.core.AttentionQuantize" \
    model.scale_factor=2
