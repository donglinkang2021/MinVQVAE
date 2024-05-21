import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *

def get_datasets():
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_dataset = CIFAR10(
        root=data_dir, 
        train=True, 
        download=False, 
        transform=transform_train
    )
    test_dataset = CIFAR10(
        root=data_dir, 
        train=False, 
        download=False, 
        transform=transform_test
    )
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
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, idxs = model(data)
        loss = criterion(output, target)
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
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, idxs = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
    metrics['val loss'] = running_loss / len(test_loader)
    return metrics

from vqvae import SubSampleBlock

class Classifier(nn.Module):
    def __init__(self, in_channel, n_classes):
        super().__init__()
        self.subsample = SubSampleBlock(in_channel, 4 * in_channel, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4 * in_channel, n_classes)

    def forward(self, x):
        # x: (B, in_channel, H, W)
        # -> (B, 4 * in_channel, H // 4, W // 4)
        x = self.subsample(x)
        # -> (B, 4 * in_channel, 1, 1)
        x = self.pool(x)
        # -> (B, 4 * in_channel)
        x = x.view(x.size(0), -1)
        # -> (B, n_classes)
        return self.fc(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = get_datasets()

    train_loader, test_loader = get_loader(train_dataset, test_dataset, 256)

    from vqvae import VQVAE
    model = VQVAE(**model_kwargs)
    model_path = "ckpt/unmask_vqvae_cifar10_10epo.pth"
    model.load_state_dict(torch.load(model_path))
    model.decoder = Classifier(in_channel=hid_channel, n_classes=10)
    model = model.to(device)

    # Freeze the VQVAE model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
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

"""
"""