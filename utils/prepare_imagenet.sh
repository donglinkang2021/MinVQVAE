cd /data1/linkdom/data/
mkdir ImageNet
src=/data1/linkdom/archive/ImageNet/ILSVRC2012
dest=/data1/linkdom/data/ImageNet/
cp -r $src/ILSVRC2012_img_train.tar $dest/
cp -r $src/ILSVRC2012_img_val.tar $dest/
cp -r $src/ILSVRC2012_devkit_t12.tar.gz $dest/