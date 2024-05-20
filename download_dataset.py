from torchvision.datasets import Places365
data_dir = '/opt/data/private/linkdom/data'
Places365(root=data_dir, split='train-standard', download=True)
Places365(root=data_dir, split='val', download=True)