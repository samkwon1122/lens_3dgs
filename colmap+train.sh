#!/bin/bash

# Usage: ./train_cam.sh <dataset> <gpu_ids>
# Usage: ./train_cam.sh GreatCourt 0,1,2,3

TARGET_DIR="/data1/heungchan/CambridgeLandmarks/$1"

CUDA_VISIBLE_DEVICES=$2 python convert.py -s $TARGET_DIR --resize
CUDA_VISIBLE_DEVICES=$2 python delete_test_seq.py --input_path $TARGET_DIR/sparse/0/images.bin
CUDA_VISIBLE_DEVICES=$2 python train.py -s $TARGET_DIR -m output/$1/seq_train -i images_3
CUDA_VISIBLE_DEVICES=$2 python render.py -m output/$1/seq_train