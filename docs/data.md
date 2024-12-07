# Data

## Prepare Dataset

> The following two datasets are relatively large, and downloading them using `torchvision.datasets` from PyTorch is too slow. Therefore, I have prepared Baidu Netdisk links for you to download.

After some thought, the first step should be to prepare the dataset. The datasets to be prepared are:

- ImageNet
- CelebA

You can use the following commands to check disk space:

```bash
df -h . # Check the disk space of the current directory
du -h --max-depth=1 . # Check the size of the folders in the current directory
```

```bash
mkdir archive # Create a folder to store data archives
```

## Download Dataset

Use bypy to download data

```shell
cd /data1/linkdom/archive 
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install bypy
bypy list
<add your access token>
```

- [ImageNet](https://pan.baidu.com/s/1WNPo2iXrDOaa_uP_Opx_qA?pwd=pqxi)
- [CelebA](https://pan.baidu.com/s/1IuvxFiy5B7T9-3B8sEemEQ?pwd=je38)

Save the data to `My Application Data\bypy\data` in Baidu Netdisk, and then use `bypy` to download the data.

```shell
tmux new -s bypy
source venv/bin/activate
bypy list data
bypy download data/ImageNet /data1/linkdom/data
bypy download data/CelebA /data1/linkdom/data
```

## Load Dataset

```shell
bash utils/prepare_celeba.sh
python utils/extract_celeba.py
bash utils/prepare_imagenet.sh
python utils/extract_imagenet.py
```
