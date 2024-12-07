import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from typing import Tuple

__all__ = ["ProductQuantize"]

class ProductQuantize(nn.Module):
    def __init__(self, v_cluster:int, g_head: int, n_embed:int, decay=0.99, eps=1e-5):
        """
        make n_embed to g_head groups, 
        and each line has n_embed // g_head dimensions (or call it head_size)
        n_embed = g_head * head_size, simplify the notation as d = gH * Hs
        """
        super().__init__()
        assert n_embed % g_head == 0, "n_embed must be divisible by g_head"
        head_size = n_embed // g_head

        self.v_cluster = v_cluster
        self.g_head = g_head
        self.n_embed = n_embed
        self.head_size = head_size

        # hyperparameters 
        # used in the update of the codebook
        self.decay = decay
        self.eps = eps

        # codebook
        # no need to learn, just initialize
        embed = torch.randn(g_head, v_cluster, head_size)
        self.register_buffer("embed", embed)                                            # [gH, v, Hs]
        self.register_buffer("cluster_size", torch.zeros(g_head, v_cluster))            # [gH, v]
        self.register_buffer("embed_avg", embed.clone())                                # [gH, v, Hs]

    def update_embed(self, input:Tensor, codes:Tensor):
        # input: [gH, B*T, Hs]
        # codes: [gH, B*T]
        
        # [gH, B*T] --onehot--> [gH, B*T, v] --transpose--> [gH, v, B*T]
        codes_onehot = F.one_hot(codes, self.v_cluster).transpose(1, 2).type(input.dtype)
        
        # [gH, v, B*T] @ [gH, B*T, Hs] -> [gH, v, Hs]
        codebook = codes_onehot @ input

        # update
        self.cluster_size = self.cluster_size * self.decay + codes_onehot.sum(-1) * (1 - self.decay)
        self.embed_avg = self.embed_avg * self.decay + codebook * (1 - self.decay)
        embed_norm = self.embed_avg / (self.cluster_size + self.eps).unsqueeze(-1)
        self.embed.data.copy_(embed_norm)

    def forward(self, input:Tensor) -> Tuple[Tensor, Tensor]:
        B, T, n_embed = input.shape
        # input: [B, T, n_embed] -> [gH, B*T, Hs]
        input = rearrange(input, "B T (gH Hs) -> gH (B T) Hs", gH=self.g_head)

        # [gH, B*T, Hs] [gH, v, Hs] -> [gH, B*T, v]
        dist = torch.cdist(input, self.embed) 

        codes = dist.argmin(-1)             # [gH, B*T]
        quantize = self.embed_code(codes)   # [gH, B*T, Hs]

        if self.training:
            self.update_embed(input, codes)
            
        # forward: quantize = quantize
        # backward: input_embed.grad = quantize.grad
        quantize = input + (quantize - input).detach()

        # [gH, B*T, Hs] -> [B, T, n_embed]
        quantize = rearrange(quantize, "gH (B T) Hs -> B T (gH Hs)", B = B)
        # [gH, B*T] -> [B, T, gH]
        codes = rearrange(codes, "gH (B T) -> B T gH", B = B)
        return quantize, codes

    def embed_code(self, embed_id: Tensor) -> Tensor:
        # embed_id: [gH, B*T]
        # [gH, B*T] --onehot--> [gH, B*T, v]
        codes_onehot = F.one_hot(embed_id, self.v_cluster).type(self.embed.dtype)
        # [gH, B*T, v] @ [gH, v, Hs] -> [gH, B*T, Hs]
        quantize = codes_onehot @ self.embed
        return quantize


if __name__ == "__main__":
    torch.manual_seed(1337)
    B = 2
    T = 100
    embd_dim = 128
    g_head = 16
    vocab_size = 32
    input = torch.randn(B, T, embd_dim, requires_grad=True)
    target = torch.randn(B, T, embd_dim)
    vq = ProductQuantize(vocab_size, g_head, embd_dim)
    quantize, idxs = vq(input)
    criterion = nn.MSELoss()
    loss = criterion(quantize, target)
    loss.backward()
    print("criterion(quantize, target)", loss.item())
    print("input.grad", input.grad.shape)
    print("criterion(quantize, input)", criterion(quantize, input).item())

    # calculate the compression ratio
    # compression ratio % version
    n_vectors = B * T
    original_size = n_vectors * embd_dim

    head_size = embd_dim // g_head
    compressed_size = g_head * vocab_size * head_size + g_head * n_vectors
    compression_ratio = compressed_size / original_size
    print(f"Compression Ratio: {compression_ratio * 100:.2f}%")

# test it with:
# python models/quantize/ProductQuantize.py
"""
criterion(quantize, target) 1.6999313831329346
input.grad torch.Size([2, 100, 128])
criterion(quantize, input) 0.637004017829895
Compression Ratio: 28.50%
"""