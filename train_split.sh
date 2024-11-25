#!/bin/bash

data=/data1/heungchan/7Scenes

python split_seq.py --input_path $data/$1/sparse/0

for seq in `ls $data/$1/sparse | grep seq*`
do
    CUDA_VISIBLE_DEVICES=$2 python train.py -s $data/$1 -m output/$1/$seq -i images_2 --seq $seq --ip 127.0.0.$(($2 + 1))
done
