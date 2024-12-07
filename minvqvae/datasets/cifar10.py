from .base import VisionDataModule, split_dataset
from torchvision.datasets import CIFAR10

class CIFAR10DataModule(VisionDataModule):
    def __init__(self, data_dir:str, batch_size:int, num_workers:int, image_size:int):
        super().__init__(data_dir, batch_size, num_workers, image_size)

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def get_datasets(self):
        entire_dataset = CIFAR10(self.data_dir, True, self.transform)
        test_set = CIFAR10(self.data_dir, False, self.transform)
        train_set, valid_set = split_dataset(entire_dataset)
        return train_set, valid_set, test_set
