"""
mask module
- `random_mask`: mask the `[B, T, C]` Tensor x on dim `T` with the specified probability
- `consecutive_mask`: mask the `[B, T, C]` Tensor x on dim `T` with the specified probability and mask length
- `patch_mask`: mask the `[B, C, H, W]` Tensor x on dim `H` and `W` with the specified probability and patch size

Note:
- `mask_prob` is the probability of masking a frame (0.0 ~ 1.0)
    - higher value means more frames are masked
"""
import torch
from torch import Tensor
from einops import rearrange

__all__ = ['random_mask', 'consecutive_mask', 'patch_mask']

def random_mask(
        x: Tensor,
        mask_prob: float,
    ) -> Tensor:
    """
    mask the `[B, T, C]` Tensor x on dim `T`
    with the specified probability
    """
    B, T, C = x.size()
    # Generate random probabilities for each frame
    rand_probs = torch.rand(B, T, device=x.device)
    mask = (rand_probs > mask_prob).unsqueeze(-1).type(x.dtype)
    x = x * mask
    return x


def consecutive_mask(
        x: Tensor, 
        mask_prob: float, 
        mask_length: int, 
    ) -> Tensor:
    """
    mask the `[B, T, C]` Tensor x on dim `T` 
    with the specified probability and mask length
    - `mask_prob` frames are randomly selected as the start of the mask
    - `mask_length` consecutive frames are masked
    """
    B, T, C = x.size()
    # T = nt * mask_length + resT
    nt = T // mask_length
    resT = T % mask_length
    mask_ones = torch.ones(B, nt, mask_length, C, device=x.device, dtype=x.dtype)
    
    # Generate random probabilities for each frame
    rand_probs = torch.rand(B, nt, device=x.device)
    
    # Create a mask based on the specified probability
    mask = (rand_probs < mask_prob).unsqueeze(-1).unsqueeze(-1)
    mask_ones.masked_fill_(mask, 0) # mask_ones[mask] = 0
    mask_ones = rearrange(mask_ones, 'b nt ts c -> b (nt ts) c')
    
    # pad the mask_ones Tensor to the same length as x
    mask_pad = torch.ones(B, resT, C, device=x.device, dtype=x.dtype)
    
    # cat [B, nt * mask_length, C] [B, resT, C] -> [B, T, C]
    mask_ones = torch.cat([mask_ones, mask_pad], dim=1)
    x = x * mask_ones
    return x

def patch_mask(
        x: Tensor,
        mask_prob: float,
        patch_size: int,
    ) -> Tensor:
    """
    mask the `[B, C, H, W]` Tensor x on dim `H` and `W`
    with the specified probability and patch size
    """
    assert x.dim() == 4, f"Expected 4D input, got {x.dim()}"
    B, C, H, W = x.size()
    assert H % patch_size == 0, f"Height {H} is not divisible by patch size {patch_size}"
    assert W % patch_size == 0, f"Width {W} is not divisible by patch size {patch_size}"
    nH, nW = H // patch_size, W // patch_size
    x = rearrange(
        x, 'B C (nH Hs) (nW Ws) -> B (nH nW) (C Hs Ws)', 
        Hs = patch_size, Ws = patch_size
    )
    x = random_mask(x, mask_prob)
    x = rearrange(
        x, 'B (nH nW) (C Hs Ws) -> B C (nH Hs) (nW Ws)', 
        Hs = patch_size, Ws = patch_size, nH = nH, nW = nW
    )
    return x


if __name__ == '__main__':
    # simple case for debugging
    B, T, C = 2, 10, 3
    x = torch.rand(B, T, C)

    mask_prob = 0.2
    mask_length = 3

    # usage example
    masked_x0 = random_mask(x, mask_prob)
    print("random mask:\n", masked_x0)
    masked_x1 = consecutive_mask(x, mask_prob, mask_length)
    print("consecutive mask:\n", masked_x1)

    B, T, C = 32, 1200, 64
    x = torch.rand(B, T, C)
    mask_prob = 0.2
    mask_length = 3
    masked_x2 = random_mask(x, mask_prob)
    masked_x3 = consecutive_mask(x, mask_prob, mask_length)
    print("random mask:\n", masked_x2[0, :20, :2])
    print("consecutive mask:\n", masked_x3[0, :20, :2])