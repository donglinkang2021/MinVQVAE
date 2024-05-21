import torch
from torch import nn
import lightning as L
from torchvision.utils import make_grid
from vqvae import VQVAE, Classifier
from sqate import SQATE
import torchmetrics
from mask import patch_mask

__all__ = [
    'VQVAELightning', 
    'SQATELightning',
    'VQVAEFinetuneLightning'
]

class VQVAELightning(L.LightningModule):
    def __init__(
            self, 
            model_kwargs:dict, 
            vis_kwargs:dict, 
            mask_kwargs, 
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
        data_masked = patch_mask(data, **self.hparams.mask_kwargs)
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

class VQVAEFinetuneLightning(L.LightningModule):
    def __init__(
            self, 
            model_path:str,
            model_kwargs:dict, 
            lr:float
        ):
        super().__init__()
        self.save_hyperparameters()
        model = VQVAE(**model_kwargs)
        model.load_state_dict(torch.load(model_path))
        model.decoder = Classifier(
            in_channel=self.hparams.model_kwargs["hid_channel"], 
            n_classes=10
        )
        # Freeze the VQVAE model
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.decoder.parameters():
        #     param.requires_grad = True
        self.vqvae = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task = "multiclass", 
            num_classes = 10
        )

    def forward(self, x):
        y, idxs = self.vqvae(x)
        return y
    
    def training_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "train_acc": accuracy}, 
                      sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss, "val_acc": accuracy}, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict({"test_loss": loss, "test_acc": accuracy}, sync_dist=True)
        return loss

    def _common_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target)
        accuracy = self.accuracy(logits, target)
        return loss, accuracy
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    