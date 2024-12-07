import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from .core.quantize import SoftQuantize
from einops import rearrange
from typing import Tuple

__all__ = [
    'VQVAE', 
    'Encoder', 
    'Decoder', 
    'ResBlock', 
    'SubSampleBlock', 
    'SubsampleTransposeBlock',
    'Classifier'
]

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
        embed_dim:int,
        n_embed:int,
        scale_factor:int
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel, 
            hid_channel, 
            n_res_block, 
            n_res_channel, 
            scale_factor
        )
        self.enc_out = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = SoftQuantize(
            vocab_size=n_embed, 
            embd_dim=embed_dim
        )
        self.dec_in = nn.Conv2d(embed_dim, hid_channel, 1)
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
        # -> enc: (B, H//2^s, W//2^s, embed_dim)        
        enc = self.enc_out(enc)
        _, _, H, W = enc.shape
        enc = rearrange(enc, 'b c h w -> b (h w) c')
        
        quant, idxs = self.quantize(enc)

        # quant: (B, H//2^s, W//2^s, embed_dim)
        # -> quant: (B, hid_channel, H//2^s, W//2^s)
        quant = rearrange(quant, 'b (h w) c -> b c h w', h=H, w=W)
        quant = self.dec_in(quant)

        # quant: (B, hid_channel, H//2^s, W//2^s)
        # -> dec: (B, in_channel, H, W)
        dec = self.decoder(quant)
        return dec, idxs

if __name__ == '__main__':
    model = VQVAE(
        in_channel=3,
        hid_channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        scale_factor=2
    )
    x = torch.randn(32, 3, 256, 256)
    y, idxs = model(x)
    print(y.shape, idxs.shape)

# python -m minvqvae.models.vqvae