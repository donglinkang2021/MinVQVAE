# Experiments

```bash
# begin to experiment
git checkout -b experiments
```

## Stage 0

总结一下前阶段做的实验：

- 之前所训练的严格意义上来说只能算作AE, 不过是添加了quantize作为中间层bottleneck的处理
    - 得到的结果就是发现目前评判quantize的metric只从mse_loss出发，不太全面
    - 最后发现还是 Identity 的效果最好 (也即最原始的 AE 的效果是最好的)
- 另外还实现了一下 Vanilla VAE，但是在 ImageNet 上的效果 reconstruction 不好
    - 没有
- 后续
    - 先在 Mnist 上跑出 Vanilla VAE 的效果
    - 应该思考的事情是先实现一下传统的vqvae，把vq_loss也加进来，考虑一下熵的问题，先实现一下原始的vqvae
    - 参考[[AntixK/PyTorch-VAE]](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py) [[google-deepmind/sonnet]](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py) [[rosinality/vq-vae-2-pytorch]](https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py) 调研一下metrics才行

```bash
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
```
