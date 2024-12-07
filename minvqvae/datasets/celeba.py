from .base import VisionDataModule
from torchvision.datasets import CelebA

class CelebADataModule(VisionDataModule):
    def __init__(self, data_dir:str, batch_size:int, num_workers:int, image_size:int):
        super().__init__(data_dir, batch_size, num_workers, image_size)

    def prepare_data(self):
        # single gpu
        CelebA(root=self.data_dir, split='train', download=True)
        CelebA(root=self.data_dir, split='valid', download=True)
        CelebA(root=self.data_dir, split='test', download=True)

    def get_datasets(self):
        train_set = CelebA(self.data_dir, 'train', transform=self.transform)
        valid_set = CelebA(self.data_dir, 'valid', transform=self.transform)
        test_set = CelebA(self.data_dir, 'test', transform=self.transform)
        return train_set, valid_set, test_set
