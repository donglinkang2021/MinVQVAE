import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from quantize import SimpleQuantize
from einops import rearrange
from typing import Tuple

__all__ = ['VQVAE']

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


class Encoder(nn.Module):
    def __init__(
            self, 
            in_channel:int, 
            hid_channel:int, 
            n_res_block:int, 
            n_res_channel:int
        ):
        super().__init__()

        blocks = [
            nn.Conv2d(
                in_channel, 
                hid_channel // 2, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hid_channel // 2, 
                hid_channel, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hid_channel, 
                hid_channel, 
                kernel_size=3, 
                stride=1,
                padding=1
            ),
        ]

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
        n_res_channel:int
    ):
        super().__init__()
        blocks = []
        for _ in range(n_res_block):
            blocks.append(ResBlock(hid_channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                nn.ConvTranspose2d(
                    hid_channel, 
                    hid_channel, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    hid_channel, 
                    hid_channel // 2, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    hid_channel // 2, 
                    out_channel, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),
            ]
        )

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
        n_embed:int
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel, 
            hid_channel, 
            n_res_block, 
            n_res_channel
        )
        self.enc_out = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = SimpleQuantize(
            vocab_size=n_embed, 
            embd_dim=embed_dim
        )
        self.dec_in = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel
        )

    def forward(self, input:Tensor) -> Tuple[Tensor, Tensor]:
        # input: (B, in_channel, H, W) 
        # -> enc: (B, hid_channel, H//4, W//4)
        enc = self.encoder(input)

        # enc: (B, hid_channel, H//4, W//4) 
        # -> enc: (B, H//4, W//4, embed_dim)        
        enc = self.enc_out(enc)
        enc = rearrange(enc, 'b c h w -> b h w c')
        
        quant, idxs = self.quantize(enc)

        # quant: (B, H//4, W//4, embed_dim)
        # -> quant: (B, hid_channel, H//4, W//4)
        quant = rearrange(quant, 'b h w c -> b c h w')
        quant = self.dec_in(quant)

        # quant: (B, hid_channel, H//4, W//4)
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
        n_embed=512
    )
    x = torch.randn(32, 3, 256, 256)
    y, idxs = model(x)
    print(y.shape)