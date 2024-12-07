# TODO

希望重新优化仓库的architecture，方便做实验和试一下不同的网络架构。

> target1: 优化仓库的architecture，重新跑起来

## 数据

### 准备数据集

> 下面这两个数据集比较大，用pytorch的`torchvision.datasets`下载太慢了，所以我准备了百度网盘的链接，可以自己下载。

思考了一下第一步应该先准备数据集，要准备的数据集有：

- ImageNet
- CelebA

可以使用命令查看磁盘空间：

```bash
df -h . # 查看当前目录所在的磁盘空间
du -h --max-depth=1 . # 查看当前目录下的文件夹的大小
```

```bash
mkdir archive # 创建一个文件夹用于存放数据压缩包
```

### 下载数据集

使用bypy下载数据

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

自己把数据都保存到百度网盘中的`我的应用数据\bypy\data`中，然后使用`bypy`下载数据。

```shell
tmux new -s bypy
source venv/bin/activate
bypy list data
bypy download data/ImageNet /data1/linkdom/data
bypy download data/CelebA /data1/linkdom/data
```

### 加载数据集

```shell
bash utils/prepare_celeba.sh
python utils/extract_celeba.py
bash utils/prepare_imagenet.sh
python utils/extract_imagenet.py
```


```bash
git checkout -b feature/prepare-dataset
git add .
```

