from torchvision.datasets import (
    MNIST, CIFAR10, CelebA, ImageNet
)

def check_MNIST(root):
    train_dataset = MNIST(root, train=True, download=True)
    test_dataset = MNIST(root, train=False, download=True)
    print(f"MNIST train dataset: {len(train_dataset)}")
    print(f"MNIST test dataset: {len(test_dataset)}")

def check_CIFAR10(root):
    train_dataset = CIFAR10(root, train=True, download=True)
    test_dataset = CIFAR10(root, train=False, download=True)
    print(f"CIFAR10 train dataset: {len(train_dataset)}")
    print(f"CIFAR10 test dataset: {len(test_dataset)}")

def check_CelebA(root):
    dataset = CelebA(root, split='all', download=True)
    print(f"CelebA dataset: {len(dataset)}")

    trainset = CelebA(root=root, split='train', download=False)
    valset = CelebA(root=root, split='valid', download=False)
    testset = CelebA(root=root, split='test', download=False)

    print(f"CelebA Train dataset: {len(trainset)}")
    print(f"CelebA Valid dataset: {len(valset)}")
    print(f"CelebA Test dataset: {len(testset)}")

def check_ImageNet(root):
    train_dataset = ImageNet(f"{root}/ImageNet", split='train')
    val_dataset = ImageNet(f"{root}/ImageNet", split='val')
    print(f"ImageNet train dataset: {len(train_dataset)}")
    print(f"ImageNet val dataset: {len(val_dataset)}")

def main():
    root = "/data1/linkdom/data"
    check_MNIST(root)
    check_CIFAR10(root)
    check_CelebA(root)
    check_ImageNet(root)

if __name__ == "__main__":
    main()

# python check_datasets.py