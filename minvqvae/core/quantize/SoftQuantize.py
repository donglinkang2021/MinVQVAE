import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

__all__ = ["SoftQuantize"]

class SoftQuantize(nn.Module):
    """softmax;embedding not updated"""
    def __init__(self, n_embed:int, embd_dim:int):
        """
        Use the softmax to replace argmax in the `SimpleQuantize`
        """
        super().__init__()
        self.embd_dim = embd_dim
        self.ln = nn.LayerNorm(embd_dim)
        self.embd = nn.Embedding(n_embed, embd_dim)

    def forward(self, input: Tensor):
        # layer norm 
        # magic code, get crazy performance
        input = self.ln(input)
        # [B, T, n_embed] @ [n_embed, n_embed] -> [B, T, n_embed]
        probs = (input @ self.embd.weight.t() / math.sqrt(self.embd_dim)).softmax(-1)
        idxs = probs.argmax(-1)
        
        quantize = probs @ self.embd.weight
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
    vq = SoftQuantize(n_embed, embd_dim)
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
# python minvqvae/core/quantize/SoftQuantize.py
"""
criterion(quantize, target) 1.056573510169983
input.grad torch.Size([2, 100, 128])
criterion(quantize, input) 0.9152400493621826
Compression Ratio: 16.78%
"""