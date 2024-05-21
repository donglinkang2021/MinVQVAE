import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *

def get_loader(data_dir:str, batch_size:int) -> DataLoader:
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    test_dataset = CIFAR10(
        root=data_dir, 
        train=False, 
        transform=transform_test, 
        download=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    return test_loader

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
    pbar = tqdm(total=len(test_loader), desc='Evaluating...')
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_loader(data_dir, 32)

    from vqvae import VQVAE
    model = VQVAE(**model_kwargs).to(device)
    model_path = "ckpt/unmask_vqvae_cifar10_10epo.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.MSELoss()
    metrics = evaluate(model, test_loader, criterion, device)
    print(metrics)

if __name__ == '__main__':
    main()

"""
{'test loss': 0.0002314429580938434}
"""