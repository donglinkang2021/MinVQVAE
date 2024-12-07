import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lightning as L
from typing import Tuple

def split_dataset(entire_dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """Splits the given dataset(train set) into training and validation sets."""
    train_set_size = int(len(entire_dataset) * 0.8)
    valid_set_size = len(entire_dataset) - train_set_size
    seed = torch.Generator().manual_seed(1337)
    return torch.utils.data.random_split(
        entire_dataset, 
        [train_set_size, valid_set_size], 
        generator = seed
    )

class VisionDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_dir:str, 
            batch_size:int, 
            num_workers:int, 
            image_size:int
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def prepare_data(self):
        """single gpu"""
        # will be implemented in subclasses
        raise NotImplementedError
    
    def get_datasets(self):
        # will be called in setup
        # will be implemented in subclasses
        raise NotImplementedError    
    
    def setup(self, stage=None):
        """multi-gpu"""
        self.train_set, self.valid_set, self.test_set = self.get_datasets()
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )
