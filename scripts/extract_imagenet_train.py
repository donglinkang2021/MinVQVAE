# for faster extraction, use ThreadPoolExecutor

import tarfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

TRAIN_SRC_DIR = '/data1/linkdom/archive/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar'
TRAIN_DEST_DIR = '/data1/linkdom/data/imagenet1/train'

def extract_train(max_workers=8):
    with open(TRAIN_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        
        def extract_class(item, pbar):
            cls_name = item.name.strip(".tar")
            a = tar.extractfile(item)
            b = tarfile.open(fileobj=a, mode="r:")
            class_path = f"{TRAIN_DEST_DIR}/{cls_name}/"
            Path(class_path).mkdir(parents=True, exist_ok=True)
            pbar.set_postfix_str(f"Extracting {cls_name}")
            b.extractall(class_path)
            pbar.update(1)
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=len(tar.getmembers()), desc='Extracting train dataset', dynamic_ncols=True)
            for item in tar:
                executor.submit(extract_class, item, pbar)
            pbar.close()


if __name__ == '__main__':
    # base line
    extract_train(16) # 137.7 GB cost 15:23
