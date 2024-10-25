#!/bin/bash

# Usage: ./train_cam.sh GreatCourt/seq1

TARGET_DIR="/data1/heungchan/CambridgeLandmarks/$1"
INPUT_DIR="$TARGET_DIR/input"

mkdir -p "$INPUT_DIR"

mv "$TARGET_DIR"/*.png "$INPUT_DIR"

python convert.py -s $TARGET_DIR --resize
python train.py -s $TARGET_DIR -r 1 -m output/$1
python render.py -m output/$1