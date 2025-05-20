import torch
from torch import nn
import lightning as L
from lightning.pytorch import loggers
from torchvision.utils import make_grid
import wandb
from .core.mask import patch_mask

class Img2Img(L.LightningModule):
    def __init__(
            self, model:nn.Module, 
            vis_kwargs:dict, 
            mask_kwargs:dict=None, 
            learning_rate:float=3e-4
        ):
        super().__init__()
        self.vis_kwargs = vis_kwargs
        self.mask_kwargs = mask_kwargs
        self.learning_rate = learning_rate
        self.model = model
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        y, idxs = self.model(x)
        return y
    
    def training_step(self, batch, batch_idx):
        loss, logits, data, data_masked = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, sync_dist=True,
                      on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, data, data_masked = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, sync_dist=True)
        self._add_image(data, 'val_original')
        self._add_image(data_masked, 'val_masked')
        self._add_image(logits, 'val_recon')
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, data, data_masked = self._common_step(batch, batch_idx)
        self.log_dict({"test_loss": loss}, sync_dist=True)
        self._add_image(data, 'test_original')
        self._add_image(data_masked, 'test_masked')
        self._add_image(logits, 'test_recon')
        return loss
    
    def _add_image(self, data:torch.Tensor, name:str):
        if data is None:
            return
        n_sample = self.vis_kwargs['n_sample']
        size = self.vis_kwargs['size']
        in_channel = self.vis_kwargs['in_channel']
        grid = make_grid(
            data[:n_sample].view(-1, in_channel, size, size)
        )
        if isinstance(self.logger, loggers.TensorBoardLogger):
            # for tensorboard
            self.logger.experiment.add_image(
                name, grid, global_step=self.global_step
            )
        elif isinstance(self.logger, loggers.WandbLogger):
            # for wandb
            self.logger.experiment.log({
                name: wandb.Image(grid, caption=name)
            }, step=self.global_step)

    def _common_step(self, batch, batch_idx):
        data, target = batch
        if self.mask_kwargs is None:
            data_masked = data
        else:
            data_masked = patch_mask(data, **self.mask_kwargs)
        logits = self(data_masked)
        loss = self.loss_fn(logits, data)
        return loss, logits, data, data_masked
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
