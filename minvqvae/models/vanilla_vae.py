import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List

class Encoder(nn.Module):
    def __init__(self, in_channel:int, hidden_dims:List[int]):
        super().__init__()
        blocks = []
        for h_dim in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, h_dim, 3, 2, 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channel = h_dim
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)
    
class Decoder(nn.Module):
    def __init__(self, hidden_dims:List[int]):
        super().__init__()
        blocks = []
        in_channel = hidden_dims[0]
        hidden_dims = hidden_dims[1:] + [hidden_dims[-1]]
        for h_dim in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channel, h_dim, 3, 2, 1, 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channel = h_dim
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)

class VanillaVAE(nn.Module):
    def __init__(self,
            in_channel: int,
            latent_dim: int,
        ) -> None:
        # assuming image size is 64x64
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        hidden_dims = [32, 64, 128, 256, 512]
        self.encoder = Encoder(in_channel, hidden_dims)
        flatten_size = hidden_dims[-1]*4
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_var = nn.Linear(flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, flatten_size)
        hidden_dims.reverse()
        self.decoder = Decoder(hidden_dims)
        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], in_channel, 3, 1, 1),
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        # input: batch of images (B,C,H,W)
        result = self.encoder(input) # (B,sC,sH,sW)
        result = torch.flatten(result, start_dim=1) # (B,CxsHxsW)
        mu = self.fc_mu(result) # (B,D)
        log_var = self.fc_var(result) # (B,D)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        # z: latent codes (B,D)
        result = self.decoder_input(z) # (B,CxsHxsW)
        result = result.view(-1, 512, 2, 2) # (B,sC,sH,sW)
        result = self.decoder(result)
        result = self.final_layer(result) # (B,C,H,W)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # mu: (B,D) logvar: (B,D)
        std = torch.exp(0.5 * logvar) # (B,D)
        eps = torch.randn_like(std) # (B,D)
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
    x = torch.randn(32, 3, 64, 64)
    model = VanillaVAE(
        in_channel=3,
        latent_dim=128
    )
    recons, loss = model(x)
    print(recons.shape) # (32, 3, 64, 64)
    print(loss) # scalar

# python -m minvqvae.models.vanilla_vae
