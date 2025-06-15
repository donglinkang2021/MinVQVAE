import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from minvqvae.models.vqvae import VQVAE
from minvqvae.core import SoftmaxQuantize

def set_seed(seed):
    """make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


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

def get_datasets(root:str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root, train=False, download=True, transform=transform)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = get_datasets("/data1/linkdom/data")
    train_loader, test_loader = get_loader(train_dataset, test_dataset, batch_size)

    model = VQVAE(
        in_channel = 1,
        hid_channel = 128,
        n_res_block = 2,
        n_res_channel = 32,
        scale_factor = 1,
        quantize = SoftmaxQuantize(
            n_embed = 512,
            embd_dim = 64
        )
    ).to(device)
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