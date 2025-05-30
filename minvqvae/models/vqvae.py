import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from einops import rearrange
from typing import Tuple

class ResBlock(nn.Module):
    def __init__(self, in_channel:int, _channel:int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel, 
                _channel, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channel, 
                in_channel, 
                kernel_size=1
            ),
        )

    def forward(self, input:Tensor) -> Tensor:
        out = self.conv(input)
        out += input
        return out

class SubSampleBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, scale_factor:int):
        """
        note that image size will be changed like this:
        (C_in, H, W) -> 
        (C , H // 2, W // 2) -> 
        (C * 2, H // 4, W // 4) -> ... 
        repeat the subsample `scale_factor` times
        and H, W should be divisible by 2^(scale_factor + 1)
        """
        super().__init__()
        hid_channel = out_channel // 2**scale_factor
        blocks = [
            nn.Conv2d(
                in_channel, hid_channel, 
                kernel_size=4, stride=2, padding=1
            ),
        ]
        for _ in range(scale_factor):
            blocks.extend([
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hid_channel, hid_channel * 2, 
                    kernel_size=4, stride=2, padding=1
                )
            ])
            hid_channel = hid_channel * 2

        blocks.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hid_channel, out_channel, 
                kernel_size=3, stride=1, padding=1
            )
        ])
        self.blocks = nn.Sequential(*blocks)


    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)

class SubsampleTransposeBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, scale_factor:int):
        super().__init__()
        hid_channel = in_channel
        blocks = [
            nn.ConvTranspose2d(
                hid_channel, hid_channel, 
                kernel_size=3, stride=1, padding=1
            ),
        ]

        for _ in range(scale_factor):
            blocks.extend([
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    hid_channel, hid_channel // 2, 
                    kernel_size=4, stride=2, padding=1
                )
            ])
            hid_channel = hid_channel // 2

        blocks.extend([
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hid_channel, out_channel, 
                kernel_size=4, stride=2, padding=1
            )
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)


class Encoder(nn.Module):
    def __init__(
            self, 
            in_channel:int, 
            hid_channel:int, 
            n_res_block:int, 
            n_res_channel:int,
            scale_factor:int
        ):
        super().__init__()

        blocks = [SubSampleBlock(in_channel, hid_channel, scale_factor)]
        for _ in range(n_res_block):
            blocks.append(ResBlock(hid_channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, 
        hid_channel:int,
        out_channel:int, 
        n_res_block:int, 
        n_res_channel:int,
        scale_factor:int
    ):
        super().__init__()
        blocks = []
        for _ in range(n_res_block):
            blocks.append(ResBlock(hid_channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(SubsampleTransposeBlock(hid_channel, out_channel, scale_factor))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel:int,
        hid_channel:int,
        n_res_block:int,
        n_res_channel:int,
        scale_factor:int,
        quantize:nn.Module
    ):
        super().__init__()
        embd_dim = quantize.embd_dim
        self.encoder = Encoder(
            in_channel, 
            hid_channel, 
            n_res_block, 
            n_res_channel, 
            scale_factor
        )
        self.enc_out = nn.Conv2d(hid_channel, embd_dim, 1)
        self.quantize = quantize
        self.dec_in = nn.Conv2d(embd_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            scale_factor
        )

    def forward(self, input:Tensor) -> Tuple[Tensor, Tensor]:
        # input: (B, in_channel, H, W) 
        # -> enc: (B, hid_channel, H//2^s, W//2^s)
        enc = self.encoder(input)

        # enc: (B, hid_channel, H//2^s, W//2^s) 
        # -> enc: (B, H//2^s, W//2^s, embd_dim)        
        enc = self.enc_out(enc)
        _, _, H, W = enc.shape
        enc = rearrange(enc, 'b c h w -> b (h w) c')
        
        quant, idxs = self.quantize(enc)

        # quant: (B, H//2^s, W//2^s, embd_dim)
        # -> quant: (B, hid_channel, H//2^s, W//2^s)
        quant = rearrange(quant, 'b (h w) c -> b c h w', h=H, w=W)
        quant = self.dec_in(quant)

        # quant: (B, hid_channel, H//2^s, W//2^s)
        # -> dec: (B, in_channel, H, W)
        recons = self.decoder(quant)
        recon_loss = F.mse_loss(recons, input)
        return recons, recon_loss

if __name__ == '__main__':
    from ..core import SoftmaxQuantize
    model = VQVAE(
        in_channel=3,
        hid_channel=128,
        n_res_block=2,
        n_res_channel=32,
        scale_factor=2,
        quantize=SoftmaxQuantize(
            n_embed=512,
            embd_dim=64
        )
    )
    x = torch.randn(32, 3, 256, 256)
    y, loss = model(x)
    print(y.shape, loss)

# python -m minvqvae.models.vqvae