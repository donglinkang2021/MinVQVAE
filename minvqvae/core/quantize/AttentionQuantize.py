import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

__all__ = ["AttentionQuantize"]

class AttentionQuantize(nn.Module):
    """softmax;embedding updated"""
    def __init__(self, n_embed:int, embd_dim:int, is_causal:bool=True):
        super().__init__()
        self.embd_dim = embd_dim
        self.is_causal = is_causal
        self.ln = nn.LayerNorm(embd_dim)
        self.embd = nn.Embedding(n_embed, embd_dim)

    def forward(self, input: Tensor):
        # [B, T, n_embed] @ [n_embed, n_embed] -> [B, T, n_embed]
        input = self.ln(input)
        quantize = F.scaled_dot_product_attention(
            query=input,
            key=self.embd.weight,
            value=self.embd.weight,
            is_causal=self.is_causal
        )
        return quantize, None


if __name__ == "__main__":
    torch.manual_seed(1337)
    B = 2
    T = 100
    embd_dim = 128
    n_embed = 32
    input = torch.randn(B, T, embd_dim, requires_grad=True)
    target = torch.randn(B, T, embd_dim)
    vq = AttentionQuantize(n_embed, embd_dim, is_causal=True)
    quantize, idxs = vq(input)
    criterion = nn.MSELoss()
    loss = criterion(quantize, target)
    loss.backward()
    print("criterion(quantize, target)", loss.item())
    print("embedding.weight.grad", vq.embd.weight.grad.shape)
    print("input.grad", input.grad.shape)
    print("criterion(quantize, input)", criterion(quantize, input).item())

    # calculate the compression ratio
    # compression ratio % version
    n_vectors = B * T
    original_size = n_vectors * embd_dim

    compressed_size = n_embed * embd_dim + n_vectors
    compression_ratio = compressed_size / original_size
    print(f"Compression Ratio: {compression_ratio * 100:.2f}%")

# test it with:
# python minvqvae/core/quantize/AttentionQuantize.py
"""
criterion(quantize, target) 1.056557297706604
embedding.weight.grad torch.Size([32, 128])
input.grad torch.Size([2, 100, 128])
criterion(quantize, input) 0.9157634973526001
Compression Ratio: 16.78%
"""