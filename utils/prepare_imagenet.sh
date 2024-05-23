cd /root/autodl-tmp/
mkdir imagenet
src=/root/autodl-pub/ImageNet/ILSVRC2012
dest=/root/autodl-tmp/imagenet
cp -r $src/ILSVRC2012_img_train.tar $dest/
cp -r $src/ILSVRC2012_img_val.tar $dest/
cp -r $src/ILSVRC2012_devkit_t12.tar.gz $dest/