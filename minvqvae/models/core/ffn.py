import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, embd_dim:int, vocab_size:int, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(embd_dim, vocab_size, bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(vocab_size, embd_dim, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class LLMFFN(nn.Module):
    def __init__(self, embd_dim:int, vocab_size:int, dropout:float=0.2, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(embd_dim, vocab_size, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(vocab_size, embd_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
