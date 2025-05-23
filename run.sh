tmux new-session -s "linkdom-vqvae"

conda activate linkdom
export CUDA_VISIBLE_DEVICES=0,1
python train.py \
    dataset=mnist \
    model.in_channel=1 \
    model.hidden_dims="[32,64,128,256]" \
    task_name=img2img-vae-mnist

python train.py \
    dataset=cifar10 \
    task_name=img2img-vae-cifar10

conda activate linkdom
export CUDA_VISIBLE_DEVICES=2,3
python train.py \
    model=vqvae \
    model.quantize._target_="minvqvae.core.Identity" \
    dataset=mnist \
    model.in_channel=1 \
    model.scale_factor=4 \
    task_name=img2img-vqvae-mnist

conda activate linkdom
export CUDA_VISIBLE_DEVICES=4,5

conda activate linkdom
export CUDA_VISIBLE_DEVICES=6,7
