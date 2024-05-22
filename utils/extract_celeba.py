import os
from torchvision.datasets import CelebA

root = '/root/autodl-tmp/'
base_folder = "celeba"

# from torchvision.datasets.utils import extract_archive
# extract_archive(os.path.join(root, base_folder, "img_align_celeba.zip"))

trainset = CelebA(root=root, split='train', download=False)
valset = CelebA(root=root, split='valid', download=False)
testset = CelebA(root=root, split='test', download=False)