import torch
import lightning as L
from light import *
from datasets import *
from config import *
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from vqvae import VQVAE

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':

    # lightning deepspeed has saved a directory instead of a file
    save_path = "logs/VQVAE_unmask_ImageNet/version_3/checkpoints/epoch=9-step=5010.ckpt"
    output_path = "ckpt/VQVAE_imagenet_lightning.pt"
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

    model = VQVAEUnmaskLightning.load_from_checkpoint(
        "ckpt/VQVAE_imagenet_lightning.pt", 
        hparams_file="logs/VQVAE_unmask_ImageNet/version_3/hparams.yaml"
    )
    vqvae = VQVAE(**model_kwargs)
    vqvae.load_state_dict(model.vqvae.state_dict())
    torch.save(vqvae.state_dict(), "ckpt/unmask_vqvae_imagenet_10epo.pth")