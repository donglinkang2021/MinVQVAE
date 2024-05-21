import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *

def get_datasets():
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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
        model:nn.Module, 
        train_loader:DataLoader, 
        optimizer:Optimizer, 
        criterion:nn.Module,
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
        pbar.set_postfix({'loss': loss.item()})
        pbar.update(1)
    pbar.close()

@torch.no_grad()
def evaluate(
        model:nn.Module, 
        test_loader:DataLoader, 
        criterion:nn.Module,
        device:torch.device
    ) -> dict:
    model.eval()
    metrics = {}
    running_loss = 0.0
    for data, _ in test_loader:
        data = data.to(device)
        output, idxs = model(data)
        loss = criterion(output, data)
        running_loss += loss.item()
    metrics['val loss'] = running_loss / len(test_loader)
    return metrics

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = get_datasets()

    train_loader, test_loader = get_loader(train_dataset, test_dataset, batch_size)

    from vqvae import VQVAE
    model = VQVAE(**model_kwargs).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pbar = tqdm(total=epochs, desc='Epochs', dynamic_ncols=True)
    for _ in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, criterion, device)
        pbar.set_postfix(metrics)
        pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    main()

# Epochs: 100%|██████| 10/10 [02:01<00:00, 12.15s/it, val loss=0.0912]