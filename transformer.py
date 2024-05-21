import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import math

__all__ = ['Transformer']

class CausalSelfAttention(nn.Module):
    """mix the head and the multi-head attention together"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        # project the queries, keys and values
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = rearrange(k, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        q = rearrange(q, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        v = rearrange(v, 'B T (nh hs) -> B nh T hs', nh=self.n_head)

        # casual self-attention: ignore "future" keys during attention
        # masked attention
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        
        # re-assemble all head outputs side by side
        y = rearrange(y, 'B nh T hs -> B T (nh hs)')
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd:int, dropout:float, bias:bool=True):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
            self, 
            n_embd:int, 
            n_head:int, 
            dropout:float
        ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class Transformer(nn.Module):

    def __init__(
            self, 
            n_embd:int, 
            n_head:int, 
            n_layer:int, 
            block_size:int, 
            dropout:float
        ):
        super().__init__()
        assert block_size is not None
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tok_emb: Tensor) -> Tensor:
        device = tok_emb.device
        B, T, D= tok_emb.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)
        # forward the model itself
        # position embeddings of shape (T, n_embd)
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        logits = self.transformer.ln_f(x) # (B, T, D)
        return logits