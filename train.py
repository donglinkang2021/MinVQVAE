import torch
from lightning import LightningModule, LightningDataModule, Trainer
import hydra

torch.set_float32_matmul_precision('medium')

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    dm:LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    model:LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    # `.fit` will save tensorboard log to a event file 
    # this file is large and we can remove it after training
    trainer.fit(model, dm)
    # `.test` save log too, different file
    trainer.test(model, dm)

if __name__ == '__main__':
    main()

# python train.py