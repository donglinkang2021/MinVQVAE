import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, ImageNet, CelebA, Places365
from torch.utils.data import DataLoader
import lightning as L

__all__ = [
    "CIFAR10DataModule",
    "MNISTDataModule",
    "ImageNetDataModule",
    "CelebADataModule",
    "Places365DataModule"
]

class CIFAR10DataModule(L.LightningDataModule):
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
        self.transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def prepare_data(self):
        # single gpu
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)
        
    
    def setup(self, stage=None):
        # multi-gpu
        entire_dataset = CIFAR10(
            root=self.data_dir, 
            train=True, 
            transform=self.transform_train, 
            download=False
        )
        self.test_set = CIFAR10(
            root=self.data_dir, 
            train=False, 
            transform=self.transform_test, 
            download=False
        )
        train_set_size = int(len(entire_dataset) * 0.8)
        valid_set_size = len(entire_dataset) - train_set_size
        seed = torch.Generator().manual_seed(1337)
        self.train_set, self.valid_set = data.random_split(
            entire_dataset, 
            [train_set_size, valid_set_size], 
            generator=seed
        )
    
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
    
class MNISTDataModule(L.LightningDataModule):
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
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def prepare_data(self):
        # single gpu
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # multi-gpu
        entire_dataset = MNIST(
            root=self.data_dir, 
            train=True, 
            transform=self.transform, 
            download=False
        )
        self.test_set = MNIST(
            root=self.data_dir, 
            train=False, 
            transform=self.transform, 
            download=False
        )
        train_set_size = int(len(entire_dataset) * 0.8)
        valid_set_size = len(entire_dataset) - train_set_size
        seed = torch.Generator().manual_seed(1337)
        self.train_set, self.valid_set = data.random_split(
            entire_dataset, 
            [train_set_size, valid_set_size], 
            generator=seed
        )

    
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
    
class ImageNetDataModule(L.LightningDataModule):
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
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    
    def prepare_data(self):
        # single gpu
        ImageNet(root=self.data_dir, split='train', download=True)
        ImageNet(root=self.data_dir, split='val', download=True)
        
    
    def setup(self, stage=None):
        # multi-gpu
        entire_dataset = ImageNet(
            root=self.data_dir, 
            split='train', 
            transform=self.transform, 
            download=False
        )
        self.test_set = ImageNet(
            root=self.data_dir, 
            split='val', 
            transform=self.transform, 
            download=False
        )
        train_set_size = int(len(entire_dataset) * 0.8)
        valid_set_size = len(entire_dataset) - train_set_size
        seed = torch.Generator().manual_seed(1337)
        self.train_set, self.valid_set = data.random_split(
            entire_dataset, 
            [train_set_size, valid_set_size], 
            generator=seed
        )
    
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
    
class CelebADataModule(L.LightningDataModule):
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
        self.transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def prepare_data(self):
        # single gpu
        CelebA(root=self.data_dir, split='train', download=True)
        CelebA(root=self.data_dir, split='valid', download=True)
        CelebA(root=self.data_dir, split='test', download=False)
        
    
    def setup(self, stage=None):
        # multi-gpu
        self.train_set = CelebA(
            root=self.data_dir, 
            split='train', 
            transform=self.transform_train, 
            download=False
        )
        self.valid_set = CelebA(
            root=self.data_dir, 
            split='valid', 
            transform=self.transform_test, 
            download=False
        )
        self.test_set = CelebA(
            root=self.data_dir, 
            split='test', 
            transform=self.transform_test, 
            download=False
        )
    
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

class Places365DataModule(L.LightningDataModule):
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
        self.transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def prepare_data(self):
        # single gpu
        Places365(root=self.data_dir, split='train-standard', download=True)
        Places365(root=self.data_dir, split='val', download=True)
        
    
    def setup(self, stage=None):
        # multi-gpu
        entire_dataset = Places365(
            root=self.data_dir, 
            split='train-standard', 
            transform=self.transform_train, 
            download=False
        )
        self.test_set = Places365(
            root=self.data_dir, 
            split='val', 
            transform=self.transform_test, 
            download=False
        )
        train_set_size = int(len(entire_dataset) * 0.8)
        valid_set_size = len(entire_dataset) - train_set_size
        seed = torch.Generator().manual_seed(1337)
        self.train_set, self.valid_set = data.random_split(
            entire_dataset, 
            [train_set_size, valid_set_size], 
            generator=seed
        )
    
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