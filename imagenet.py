from torchvision.datasets import ImageNet
from torchvision.datasets.imagenet import parse_val_archive

data_dir = "/root/autodl-tmp/imagenet/"
val_dataset = ImageNet(data_dir, split='val')