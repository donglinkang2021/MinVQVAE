from torchvision.datasets import CelebA
data_dir = '/opt/data/private/linkdom/data'
CelebA(root=data_dir, split='train', download=True)
CelebA(root=data_dir, split='val', download=True)