# TODO

希望重新优化仓库的architecture，方便做实验和试一下不同的网络架构。

## 优化仓库的architecture，重新跑起来

### 准备数据集

思考了一下第一步应该先准备数据集，要准备的数据集有：

- ImageNet
- CelebA
- CIFAR10

可以使用命令查看磁盘空间：

```bash
df -h . # 查看当前目录所在的磁盘空间
du -h --max-depth=1 . # 查看当前目录下的文件夹的大小
```

```bash
mkdir archive # 创建一个文件夹用于存放数据压缩包
```

下载链接

- [ImageNet](https://pan.baidu.com/s/1WNPo2iXrDOaa_uP_Opx_qA?pwd=pqxi)
- [CelebA](https://pan.baidu.com/s/1IuvxFiy5B7T9-3B8sEemEQ?pwd=je38)

```bash
git checkout -b feature/prepare-dataset
git add .
```
