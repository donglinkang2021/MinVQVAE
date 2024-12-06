from torchvision.datasets import ImageNet

data_dir = "/data1/linkdom/data/ImageNet/"
val_dataset = ImageNet(data_dir, split='val')
train_dataset = ImageNet(data_dir, split='train')
print(f"ImageNet val dataset: {len(val_dataset)}")
print(f"ImageNet train dataset: {len(train_dataset)}")

# bash utils/prepare_imagenet.sh
# python utils/extract_imagenet.py