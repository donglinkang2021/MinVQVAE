from .base import VisionDataModule, split_dataset
from torchvision.datasets import ImageNet

class ImageNetDataModule(VisionDataModule):
    def __init__(self, data_dir:str, batch_size:int, num_workers:int, image_size:int):
        super().__init__(data_dir, batch_size, num_workers, image_size)

    def prepare_data(self):
        ImageNet(root=self.data_dir, split='train')
        ImageNet(root=self.data_dir, split='val')

    def get_datasets(self):
        entire_dataset = ImageNet(self.data_dir, 'train', transform=self.transform)
        test_set = ImageNet(self.data_dir, 'val', transform=self.transform)
        train_set, valid_set = split_dataset(entire_dataset)
        return train_set, valid_set, test_set
