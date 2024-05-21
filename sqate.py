from vqvae import VQVAE
from quantize import SoftQuantize
from transformer import Transformer
from torch import Tensor
from torch import nn
from einops import rearrange

__all__ = ['SQATE']

class SQATE(nn.Module):
    def __init__(
        self,
        vqvae_kwargs:dict,
        transformer_kwargs:dict
    ):
        super().__init__()
        self.vqvae = VQVAE(**vqvae_kwargs)
        self.transformer = Transformer(**transformer_kwargs)
        
    def forward(self, input:Tensor) -> Tensor:
        # input: (B, in_channel, H, W) 
        # -> enc: (B, hid_channel, H//2^s, W//2^s)
        enc = self.vqvae.encoder(input)

        # enc: (B, hid_channel, H//2^s, W//2^s) 
        # -> enc: (B, H//2^s, W//2^s, embed_dim)        
        enc = self.vqvae.enc_out(enc)
        _, _, H, W = enc.shape
        enc = rearrange(enc, 'b c h w -> b (h w) c')
        
        quant, idxs = self.vqvae.quantize(enc)
        quant = self.transformer(quant) # just one line difference 

        # quant: (B, H//2^s, W//2^s, embed_dim)
        # -> quant: (B, hid_channel, H//2^s, W//2^s)
        quant = rearrange(quant, 'b (h w) c -> b c h w', h=H, w=W)
        quant = self.vqvae.dec_in(quant)

        # quant: (B, hid_channel, H//2^s, W//2^s)
        # -> dec: (B, in_channel, H, W)
        dec = self.vqvae.decoder(quant)
        return dec, idxs