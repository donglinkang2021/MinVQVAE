import os

# extract
root = '/data1/linkdom/data/'
base_folder = 'celeba'
from torchvision.datasets.utils import extract_archive
extract_archive(os.path.join(root, base_folder, "img_align_celeba.zip"))

# load dataset
from torchvision.datasets import CelebA

trainset = CelebA(root=root, split='train', download=False)
valset = CelebA(root=root, split='valid', download=False)
testset = CelebA(root=root, split='test', download=False)

print(f"Train dataset: {len(trainset)}")
print(f"Valid dataset: {len(valset)}")
print(f"Test dataset: {len(testset)}")

# bash utils/prepare_celeba.sh
# python utils/extract_celeba.py