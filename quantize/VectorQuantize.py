import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["VectorQuantize"]

class VectorQuantize(nn.Module):
    def __init__(self, v_cluster:int, n_embed:int, decay=0.99, eps=1e-5):
        super().__init__()

        self.v_cluster = v_cluster
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # codebook
        # no need to learn, just initialize
        embed = torch.randn(v_cluster, n_embed)
        self.register_buffer("embed", embed)                            # [v_cluster, n_embed]
        self.register_buffer("cluster_size", torch.zeros(v_cluster))    # [v_cluster]
        self.register_buffer("embed_avg", embed.clone())                # [v_cluster, n_embed]

    def update_embed(self, input:Tensor, codes:Tensor):
        # input: [B, T, n_embed]
        # codes: [B, T]

        # this implementation is not memory efficient
        # flatten = input.reshape(-1, self.n_embed)                                       # [B*T, n_embed]
        # codes_onehot = F.one_hot(codes.reshape(-1), self.v_cluster).type(flatten.dtype) # [B*T, v_cluster]
        # codebook = codes_onehot.transpose(0, 1) @ flatten                               # [v_cluster, n_embed]

        # this implementation is memory efficient
        codes_onehot = F.one_hot(codes, self.v_cluster).type(input.dtype)               # [B, T, v_cluster]
        input = input.permute(1, 0, 2)                                                  # [T, B, n_embed]
        codes_onehot = codes_onehot.permute(1, 0, 2)                                    # [T, B, v_cluster]
        codebook = codes_onehot.transpose(1, 2) @ input                                 # [T, v_cluster, n_embed]
        codebook = codebook.mean(0)
        codes_onehot = codes_onehot.reshape(-1, self.v_cluster)                         # [B*T, v_cluster]

        # update
        self.cluster_size = self.cluster_size * self.decay + codes_onehot.sum(0) * (1 - self.decay)
        self.embed_avg = self.embed_avg * self.decay + codebook * (1 - self.decay)
        embed_norm = self.embed_avg / (self.cluster_size + self.eps).unsqueeze(1)
        self.embed.data.copy_(embed_norm)

    def forward(self, input):
        # input: [B, T, n_embed]
        # [B, T, n_embed] @ [v_cluster, n_embed] -> [B, T, v_cluster]
        dist = torch.cdist(input, self.embed) 

        codes = dist.argmin(-1)             # [B, T]
        quantize = self.embed_code(codes)   # [B, T, n_embed]

        if self.training:
            self.update_embed(input, codes)
            
        # forward: quantize = quantize
        # backward: input_embed.grad = quantize.grad
        quantize = input + (quantize - input).detach()
        return quantize, codes

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed)


if __name__ == "__main__":
    torch.manual_seed(1337)
    B = 2
    T = 100
    embd_dim = 128
    vocab_size = 32
    input = torch.randn(B, T, embd_dim, requires_grad=True)
    target = torch.randn(B, T, embd_dim)
    vq = VectorQuantize(vocab_size, embd_dim)
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

    compressed_size = vocab_size * embd_dim + n_vectors
    compression_ratio = compressed_size / original_size
    print(f"Compression Ratio: {compression_ratio * 100:.2f}%")

# test it with:
# python models/quantize/VectorQuantize.py
"""
criterion(quantize, target) 1.8839858770370483
input.grad torch.Size([2, 100, 128])
criterion(quantize, input) 1.6130305528640747
Compression Ratio: 16.78%
"""