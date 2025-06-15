cd /data1/linkdom/data/
mkdir celeba
src=/data1/linkdom/archive/CelebA
tgt=/data1/linkdom/data/celeba
cp $src/Anno/*.txt $tgt
cp $src/Eval/*.txt $tgt
cp $src/Img/img_align_celeba.zip $tgt