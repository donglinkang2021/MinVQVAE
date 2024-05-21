import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from light import *
from datasets import *
from config import *
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from vqvae import VQVAE

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':

    # lightning deepspeed has saved a directory instead of a file
    save_path = "logs/VQVAE_finetune_CIFAR10/version_1/checkpoints/epoch=9-step=790.ckpt"
    output_path = "ckpt/VQVAE_finetune_CIFAR10_lightning.pt"
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

    # model = VQVAELightning.load_from_checkpoint(
    #     "ckpt/VQVAE_CIFAR10.pt", 
    #     hparams_file="logs/VQVAE_CIFAR10/version_8/hparams.yaml"
    # )
    # vqvae = VQVAE(**model_kwargs)
    # vqvae.load_state_dict(model.vqvae.state_dict())
    # torch.save(vqvae.state_dict(), "ckpt/unmask_vqvae_cifar10_10epo.pth")