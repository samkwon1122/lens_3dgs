#!/bin/bash


for i in 01 02 04 06; do python novel_view_7scene.py --input_path /data1/heungchan/7Scenes/chess/sparse/seq-${i}; done

for i in 01 02; do python novel_view_7scene.py --input_path /data1/heungchan/7Scenes/fire/sparse/seq-${i}; done

for i in 02; do python novel_view_7scene.py --input_path /data1/heungchan/7Scenes/heads/sparse/seq-${i}; done

for i in 01 03 04 05 08 10; do python novel_view_7scene.py --input_path /data1/heungchan/7Scenes/office/sparse/seq-${i}; done

for i in 02 03 06 08; do python novel_view_7scene.py --input_path /data1/heungchan/7Scenes/pumpkin/sparse/seq-${i}; done

for i in 01 02 05 07 08 11 13; do python novel_view_7scene.py --input_path /data1/heungchan/7Scenes/redkitchen/sparse/seq-${i}; done

for i in 02 03 05 06; do python novel_view_7scene.py --input_path /data1/heungchan/7Scenes/stairs/sparse/seq-${i}; done
