import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from light import *
from datasets import *
from config import *

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':

    if dataset_name == 'CIFAR10':
        dm = CIFAR10DataModule(**dataset_kwargs)
    elif dataset_name == 'MNIST':
        dm = MNISTDataModule(**dataset_kwargs)

    if model_name == 'VQVAE':
        model = VQVAELightning(
            model_kwargs, 
            vis_kwargs, 
            mask_kwargs, 
            learning_rate
        )
    elif model_name == 'SQATE':
        model = SQATELightning(
            model_kwargs, 
            transformer_kwargs, 
            vis_kwargs, 
            mask_kwargs, 
            learning_rate
        )
    elif model_name == 'VQVAE_finetune':
        # model = VQVAEFinetuneLightning(
        #     model_path = "ckpt/unmask_vqvae_cifar10_10epo.pth", 
        #     model_kwargs = model_kwargs, 
        #     lr = learning_rate
        # )
        model = VQVAEFinetuneLightning.load_from_checkpoint(
            "ckpt/VQVAE_finetune_CIFAR10_lightning.pt",
            model_path = "ckpt/unmask_vqvae_cifar10_10epo.pth", 
            model_kwargs = model_kwargs, 
            lr = learning_rate
        )

    logger = TensorBoardLogger("logs", name=f"{model_name}_{dataset_name}")
    
    trainer = L.Trainer(
        accelerator="gpu",
        strategy=DeepSpeedStrategy(),
        devices=[0, 1],
        # precision="16-mixed",
        precision=32,
        logger=logger,
        num_nodes=1,
        max_epochs=epochs,
        profiler="simple"
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)