from .base import VisionDataModule, split_dataset
from torchvision.datasets import MNIST

class MNISTDataModule(VisionDataModule):
    def __init__(self, data_dir:str, batch_size:int, num_workers:int, image_size:int):
        super().__init__(data_dir, batch_size, num_workers, image_size)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def get_datasets(self):
        entire_dataset = MNIST(self.data_dir, True, self.transform)
        test_set = MNIST(self.data_dir, False, self.transform)
        train_set, valid_set = split_dataset(entire_dataset)
        return train_set, valid_set, test_set
