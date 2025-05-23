import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["KmeansQuantize"]

class KmeansQuantize(nn.Module):
    def __init__(self, n_embed:int, embd_dim:int):
        super().__init__()
        self.embd_dim = embd_dim
        self.register_buffer("embd", torch.randn(n_embed, embd_dim))

    def forward(self, input: Tensor):
        # [B, T, embd_dim] @ [embd_dim, n_embed] -> [B, T, n_embed]
        similarity = input @ self.embd.t() # [B, T, n_embed]
        idxs = similarity.argmax(-1) # [B, T]
        quantize = self.embed_code(idxs)

        if self.training:
            # update the codebook
            B, T, V = similarity.size()
            similarity = similarity.reshape(-1, V).transpose(0, 1) # [n_embed, B*T]
            vocab_idx = similarity.argmax(-1) # [n_embed]
            embd = F.embedding(vocab_idx, input.reshape(-1, self.embd_dim)) # [n_embed, embd_dim]
            self.embd.data.copy_(embd)

        quantize = input + (quantize - input).detach()
        return quantize, idxs
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embd)


if __name__ == "__main__":
    torch.manual_seed(1337)
    B = 2
    T = 100
    embd_dim = 128
    n_embed = 32
    input = torch.randn(B, T, embd_dim, requires_grad=True)
    target = torch.randn(B, T, embd_dim)
    vq = KmeansQuantize(n_embed, embd_dim)
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

    compressed_size = n_embed * embd_dim + n_vectors
    compression_ratio = compressed_size / original_size
    print(f"Compression Ratio: {compression_ratio * 100:.2f}%")

# test it with:
# python quantize/KmeansQuantize.py
"""
criterion(quantize, target) 2.0421433448791504
input.grad torch.Size([2, 100, 128])
criterion(quantize, input) 1.685978651046753
Compression Ratio: 16.78%
"""