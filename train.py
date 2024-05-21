import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from light import VQVAELightning
from datasets import *
from config import *

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':

    if dataset_name == 'CIFAR10':
        dm = CIFAR10DataModule(**dataset_kwargs)
    elif dataset_name == 'MNIST':
        dm = MNISTDataModule(**dataset_kwargs)

    model = VQVAELightning(model_kwargs, vis_kwargs, mask_kwargs, learning_rate)

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