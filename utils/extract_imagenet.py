import os
import tarfile
from tqdm import tqdm

TRAIN_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar'
TRAIN_DEST_DIR = '/root/autodl-tmp/imagenet/train'
VAL_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar'
VAL_DEST_DIR = '/root/autodl-tmp/imagenet/val'
TEST_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_test.tar'
TEST_DEST_DIR = '/root/autodl-tmp/imagenet/test'

def extract_train():
    with open(TRAIN_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        total_files = len(tar.getmembers())
        pbar = tqdm(total=total_files, desc='Extracting train dataset', dynamic_ncols=True)
        for item in tar:
            cls_name = item.name.strip(".tar")
            a = tar.extractfile(item)
            b = tarfile.open(fileobj=a, mode="r:")
            class_path = f"{TRAIN_DEST_DIR}/{cls_name}/"
            if not os.path.isdir(class_path):
                os.makedirs(class_path)
            pbar.set_postfix({">>> ",class_path})
            b.extractall(class_path)
            pbar.update(1)
        pbar.close()


def extract_val(src:str, dest:str):
    with open(src, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        if not os.path.isdir(dest):
            os.makedirs(dest)
        print("extract val dateset to >>>", dest)
        names = tar.getnames()
        for name in names:
            tar.extract(name, dest)


if __name__ == '__main__':
    extract_train() # 137.7 GB
    extract_val(VAL_SRC_DIR, VAL_DEST_DIR) # 6.28 GB
    extract_val(TEST_SRC_DIR, TEST_DEST_DIR) # 12.74 GB