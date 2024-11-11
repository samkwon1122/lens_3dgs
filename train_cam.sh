#!/bin/bash

# Usage: ./train_cam.sh <dataset>/<seq> <gpu_ids> <ip>
# Usage: ./train_cam.sh GreatCourt/seq1 0,1,2,3 1

TARGET_DIR="/data1/heungchan/CambridgeLandmarks/$1"
INPUT_DIR="$TARGET_DIR/input"

mkdir -p "$INPUT_DIR"

mv "$TARGET_DIR"/*.png "$INPUT_DIR"

CUDA_VISIBLE_DEVICES=$2 python convert.py -s $TARGET_DIR

if [ -d "$TARGET_DIR/distorted/sparse/1" ]; then
    mv "$TARGET_DIR/distorted/sparse/0" "$TARGET_DIR/distorted/sparse/2"
    mv "$TARGET_DIR/distorted/sparse/1" "$TARGET_DIR/distorted/sparse/0"
    rm -rf "$TARGET_DIR/images"
    rm -rf "$TARGET_DIR/sparse"
    rm -rf "$TARGET_DIR/stereo"
    rm "run-colmap*.sh"
fi

CUDA_VISIBLE_DEVICES=$2 python convert.py -s $TARGET_DIR --skip_matching --resize
CUDA_VISIBLE_DEVICES=$2 python train.py -s $TARGET_DIR -r 1 -m output/$1 --ip 127.0.0.$3
CUDA_VISIBLE_DEVICES=$2 python render.py -m output/$1