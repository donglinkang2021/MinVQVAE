from torchvision.datasets.imagenet import (
    parse_devkit_archive,
    parse_train_archive,
    parse_val_archive
)

data_dir = "/root/autodl-tmp/imagenet/"

# Extract the ImageNet archive
parse_devkit_archive(data_dir)
parse_val_archive(data_dir)
# parse_train_archive(data_dir)