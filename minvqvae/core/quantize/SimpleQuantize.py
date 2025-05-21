import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SimpleQuantize"]

class SimpleQuantize(nn.Module):
    """argmax;embedding not updated"""
    def __init__(self, n_embed:int, embd_dim:int):
        super().__init__()
        self.embd_dim = embd_dim
        self.ln = nn.LayerNorm(embd_dim)
        self.embd = nn.Embedding(n_embed, embd_dim)

    def forward(self, input: Tensor):
        # [B, T, n_embed] @ [n_embed, n_embed] -> [B, T, n_embed]
        input = self.ln(input)
        idxs = (input @ self.embd.weight.t()).argmax(-1) # [B, T]
        quantize = self.embd(idxs)
        quantize = input + (quantize - input).detach()
        return quantize, idxs


if __name__ == "__main__":
    torch.manual_seed(1337)
    B = 2
    T = 100
    embd_dim = 128
    n_embed = 32
    input = torch.randn(B, T, embd_dim, requires_grad=True)
    target = torch.randn(B, T, embd_dim)
    vq = SimpleQuantize(n_embed, embd_dim)
    quantize, idxs = vq(input)
    criterion = nn.MSELoss()
    loss = criterion(quantize, target)
    loss.backward()
    print("criterion(quantize, target)", loss.item())
    print("embedding.weight.grad", vq.embd.weight.grad)
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
# python minvqvae/core/quantize/SimpleQuantize.py
"""
criterion(quantize, target) 2.0421433448791504
input.grad torch.Size([2, 100, 128])
criterion(quantize, input) 1.685978651046753
Compression Ratio: 16.78%
"""