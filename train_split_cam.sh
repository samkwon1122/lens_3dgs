#!/bin/bash

data=/data1/heungchan/CambridgeLandmarks

for seq in `ls $data/$1/sparse | grep seq*`
do
    CUDA_VISIBLE_DEVICES=$2 python train.py -s $data/$1 -m /data1/heungchan/output/$1/$seq -i images_3 --seq $seq --ip 127.0.0.$(($2 + 1))
done
