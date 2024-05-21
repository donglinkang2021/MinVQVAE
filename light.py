import torch
from torch import nn
import lightning as L
from torchvision.utils import make_grid
from vqvae import VQVAE
from sqate import SQATE
from mask import patch_mask

__all__ = ['VQVAELightning', 'SQATELightning']

class VQVAELightning(L.LightningModule):
    def __init__(
            self, 
            model_kwargs:dict, 
            vis_kwargs:dict, 
            # mask_kwargs, 
            lr:float
        ):
        super().__init__()
        self.save_hyperparameters()
        self.vqvae = VQVAE(**model_kwargs)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        y, idxs = self.vqvae(x)
        return y
    
    def training_step(self, batch, batch_idx):
        loss, logits, data, data_masked = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, sync_dist=True,
                      on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 100 == 0:
            self._add_image(data, 'train_original')
            self._add_image(data_masked, 'train_masked')
            self._add_image(logits, 'train_recon')

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
        n_sample = self.hparams.vis_kwargs['n_sample']
        size = self.hparams.vis_kwargs['size']
        in_channel = self.hparams.vis_kwargs['in_channel']
        grid = make_grid(
            data[:n_sample].view(-1, in_channel, size, size)
        )
        self.logger.experiment.add_image(
            name, grid, global_step=self.global_step
        )

    def _common_step(self, batch, batch_idx):
        data, target = batch
        # data_masked = patch_mask(data, **self.hparams.mask_kwargs)
        data_masked = None
        logits = self(data_masked)
        loss = self.loss_fn(logits, data)
        return loss, logits, data, data_masked
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    

class SQATELightning(L.LightningModule):
    def __init__(
            self, 
            model_kwargs:dict, 
            transformer_kwargs:dict, 
            vis_kwargs:dict, 
            mask_kwargs, 
            lr:float
        ):
        super().__init__()
        self.save_hyperparameters()
        self.sqate = SQATE(model_kwargs, transformer_kwargs)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        y, idxs = self.sqate(x)
        return y
    
    def training_step(self, batch, batch_idx):
        loss, logits, data, data_masked = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, sync_dist=True,
                      on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 100 == 0:
            self._add_image(data, 'train_original')
            self._add_image(data_masked, 'train_masked')
            self._add_image(logits, 'train_recon')

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
        n_sample = self.hparams.vis_kwargs['n_sample']
        size = self.hparams.vis_kwargs['size']
        in_channel = self.hparams.vis_kwargs['in_channel']
        grid = make_grid(
            data[:n_sample].view(-1, in_channel, size, size)
        )
        self.logger.experiment.add_image(
            name, grid, global_step=self.global_step
        )

    def _common_step(self, batch, batch_idx):
        data, target = batch
        data_masked = patch_mask(data, **self.hparams.mask_kwargs)
        logits = self(data_masked)
        loss = self.loss_fn(logits, data)
        return loss, logits, data, data_masked
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer