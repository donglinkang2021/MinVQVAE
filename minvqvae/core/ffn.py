import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, embd_dim:int, n_embed:int, bias=True):
        super().__init__()
        self.embd_dim = embd_dim
        self.c_fc    = nn.Linear(embd_dim, n_embed, bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(n_embed, embd_dim, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x, None

class LLMFFN(nn.Module):
    def __init__(self, embd_dim:int, n_embed:int, dropout:float=0.2, bias=True):
        super().__init__()
        self.embd_dim = embd_dim
        self.c_fc    = nn.Linear(embd_dim, n_embed, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(n_embed, embd_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, None
