import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from typing import Tuple
import math

def set_seed(seed):
    """make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -----------------------------------core-----------------------------------

class SoftQuantize(nn.Module):
    def __init__(self, vocab_size:int, embd_dim:int):
        """
        Use the softmax to replace argmax in the `SimpleQuantize`
        """
        super().__init__()
        self.embd_dim = embd_dim
        self.ln = nn.LayerNorm(embd_dim)
        self.embd = nn.Embedding(vocab_size, embd_dim)

    def forward(self, input: Tensor):
        # layer norm 
        # magic code, get crazy performance
        input = self.ln(input)
        # [B, T, n_embed] @ [n_embed, vocab_size] -> [B, T, vocab_size]
        probs = (input @ self.embd.weight.t() / math.sqrt(self.embd_dim)).softmax(-1)
        idxs = probs.argmax(-1)
        
        quantize = probs @ self.embd.weight
        quantize = input + (quantize - input).detach()
        return quantize, idxs


# -----------------------------------model-----------------------------------


class ResBlock(nn.Module):
    def __init__(self, in_channel:int, _channel:int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channel, 
                _channel, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channel, 
                in_channel, 
                kernel_size=1
            ),
        )

    def forward(self, input:Tensor) -> Tensor:
        out = self.conv(input)
        out += input
        return out

class SubSampleBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, scale_factor:int):
        """
        note that image size will be changed like this:
        (C_in, H, W) -> 
        (C , H // 2, W // 2) -> 
        (C * 2, H // 4, W // 4) -> ... 
        repeat the subsample `scale_factor` times
        and H, W should be divisible by 2^(scale_factor + 1)
        """
        super().__init__()
        hid_channel = out_channel // 2**scale_factor
        blocks = [
            nn.Conv2d(
                in_channel, hid_channel, 
                kernel_size=4, stride=2, padding=1
            ),
        ]
        for _ in range(scale_factor):
            blocks.extend([
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hid_channel, hid_channel * 2, 
                    kernel_size=4, stride=2, padding=1
                )
            ])
            hid_channel = hid_channel * 2

        blocks.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hid_channel, out_channel, 
                kernel_size=3, stride=1, padding=1
            )
        ])
        self.blocks = nn.Sequential(*blocks)


    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)

class SubsampleTransposeBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, scale_factor:int):
        super().__init__()
        hid_channel = in_channel
        blocks = [
            nn.ConvTranspose2d(
                hid_channel, hid_channel, 
                kernel_size=3, stride=1, padding=1
            ),
        ]

        for _ in range(scale_factor):
            blocks.extend([
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    hid_channel, hid_channel // 2, 
                    kernel_size=4, stride=2, padding=1
                )
            ])
            hid_channel = hid_channel // 2

        blocks.extend([
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hid_channel, out_channel, 
                kernel_size=4, stride=2, padding=1
            )
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)


class Encoder(nn.Module):
    def __init__(
            self, 
            in_channel:int, 
            hid_channel:int, 
            n_res_block:int, 
            n_res_channel:int,
            scale_factor:int
        ):
        super().__init__()

        blocks = [SubSampleBlock(in_channel, hid_channel, scale_factor)]
        for _ in range(n_res_block):
            blocks.append(ResBlock(hid_channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, 
        hid_channel:int,
        out_channel:int, 
        n_res_block:int, 
        n_res_channel:int,
        scale_factor:int
    ):
        super().__init__()
        blocks = []
        for _ in range(n_res_block):
            blocks.append(ResBlock(hid_channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(SubsampleTransposeBlock(hid_channel, out_channel, scale_factor))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel:int,
        hid_channel:int,
        n_res_block:int,
        n_res_channel:int,
        embed_dim:int,
        n_embed:int,
        scale_factor:int
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel, 
            hid_channel, 
            n_res_block, 
            n_res_channel, 
            scale_factor
        )
        self.enc_out = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = SoftQuantize(
            vocab_size=n_embed, 
            embd_dim=embed_dim
        )
        self.dec_in = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            scale_factor
        )

    def forward(self, input:Tensor) -> Tuple[Tensor, Tensor]:
        # input: (B, in_channel, H, W) 
        # -> enc: (B, hid_channel, H//2^s, W//2^s)
        enc = self.encoder(input)

        # enc: (B, hid_channel, H//2^s, W//2^s) 
        # -> enc: (B, H//2^s, W//2^s, embed_dim)        
        enc = self.enc_out(enc)
        _, _, H, W = enc.shape
        enc = rearrange(enc, 'b c h w -> b (h w) c')
        
        quant, idxs = self.quantize(enc)

        # quant: (B, H//2^s, W//2^s, embed_dim)
        # -> quant: (B, hid_channel, H//2^s, W//2^s)
        quant = rearrange(quant, 'b (h w) c -> b c h w', h=H, w=W)
        quant = self.dec_in(quant)

        # quant: (B, hid_channel, H//2^s, W//2^s)
        # -> dec: (B, in_channel, H, W)
        dec = self.decoder(quant)
        return dec, idxs

# -----------------------------------utils-----------------------------------
# future work: try this function
def init_weights(m):
    # to use this function, you should call `model.apply(init_weights)`
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0, 1)

# -----------------------------------train-----------------------------------

def get_datasets():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_loader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(
        model:torch.nn.Module, 
        train_loader:DataLoader, 
        optimizer:torch.optim.Optimizer, 
        criterion:torch.nn.Module,
        device:torch.device
    ) -> None:
    model.train()
    pbar = tqdm(total=len(train_loader), desc='Training', dynamic_ncols=True, leave=False)
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, idxs = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        pbar.update(1)
    pbar.close()

@torch.no_grad()
def evaluate(
        model:torch.nn.Module, 
        test_loader:DataLoader, 
        criterion:torch.nn.Module,
        device:torch.device
    ) -> dict:
    model.eval()
    metrics = {}
    running_loss = 0.0
    pbar = tqdm(total=len(test_loader), desc='Evaluating', dynamic_ncols=True)
    for data, _ in test_loader:
        data = data.to(device)
        output, idxs = model(data)
        loss = criterion(output, data)
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        pbar.update(1)
    pbar.close()
    metrics['test loss'] = running_loss / len(test_loader)
    return metrics

# -----------------------------------main-----------------------------------

def main():

    # training config
    batch_size = 512
    learning_rate = 3e-4
    epochs = 10

    # model config
    model_args = dict(
        in_channel = 1,
        hid_channel = 128,
        n_res_block = 2,
        n_res_channel = 32,
        embed_dim = 64,
        n_embed = 512,
        scale_factor = 1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = get_datasets()
    train_loader, test_loader = get_loader(train_dataset, test_dataset, batch_size)

    model = VQVAE(**model_args).to(device)
    # model.apply(init_weights) # future work
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pbar = tqdm(total=epochs, desc='Epochs', dynamic_ncols=True)
    for _ in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, criterion, device)
        pbar.set_postfix(metrics)
        pbar.update(1)
    pbar.close()

    model_path = "ckpt/vqvae_mnist_10epo.pth"
    torch.save(model.state_dict(), model_path)

    # model = VQVAE(**model_args).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # metrics = evaluate(model, test_loader, criterion, device)
    # print(metrics)

if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=7 python utils/train_eval_scratch.py
# Epochs: 100%|██████| 10/10 [02:01<00:00, 12.15s/it, val loss=0.0912]