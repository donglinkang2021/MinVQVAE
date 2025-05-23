import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, in_channel:int, hidden_dims:List[int]):
        super().__init__()
        blocks = []
        for h_dim in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, h_dim, 3, 2, 1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channel = h_dim
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)
    
class Decoder(nn.Module):
    def __init__(self, hidden_dims:List[int], out_channel:int):
        super().__init__()
        blocks = []
        in_channel = hidden_dims[0]
        hidden_dims = hidden_dims[1:]
        for h_dim in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channel, h_dim, 3, 2, 1, 1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channel = h_dim
        blocks.append(
            nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1)
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)

class VanillaVAE(nn.Module):
    def __init__(self,
            in_channel: int,
            latent_dim: int,
            img_size:int = 256,
            hidden_dims:List[int] = [32, 64, 128, 256, 512]
        ) -> None:
        # assuming image size is img_size x img_size 
        self.feat_size = img_size // (2**len(hidden_dims))
        feat_channel = hidden_dims[-1]
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.feat_channel = feat_channel
        self.encoder = Encoder(in_channel, hidden_dims)
        self.fc_mu = nn.Linear(feat_channel, latent_dim)
        self.fc_var = nn.Linear(feat_channel, latent_dim)
        self.dec_in = nn.Linear(latent_dim, feat_channel)
        hidden_dims.reverse()
        self.decoder = Decoder(hidden_dims, in_channel)

    def encode(self, input: Tensor) -> List[Tensor]:
        # input: batch of images (B,C,H,W)
        z = self.encoder(input) # (B,fC,fH,fW)
        z = rearrange(z, 'b c h w -> b h w c') # (B,fH,fW,fC)
        mu = self.fc_mu(z) # (B,fH,fW,D)
        log_var = self.fc_var(z) # (B,fH,fW,D)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        # z: latent codes (B,fH,fW,D)
        result = self.dec_in(z) # (B,fCxfHxfW)
        result = result.view(-1, self.feat_channel, self.feat_size, self.feat_size) # (B,fC,fH,fW)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # mu: (B,fH,fW,D) logvar: (B,fH,fW,D)
        std = torch.exp(0.5 * logvar) # (B,fH,fW,D)
        eps = torch.randn_like(std) # (B,fH,fW,D)
        return eps * std + mu

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        recons_loss = F.mse_loss(recons, input)
        kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - log_var - 1, dim=1).mean()
        loss = recons_loss + kl_loss
        return recons, loss

if __name__ == '__main__':
    img_size = 256
    x = torch.randn(32, 3, img_size, img_size)
    model = VanillaVAE(
        in_channel=3,
        latent_dim=128,
        img_size=img_size
    )
    recons, loss = model(x)
    print(recons.shape) # (32, 3, img_size, img_size)
    print(loss) # scalar

# python -m minvqvae.models.vanilla_vae
