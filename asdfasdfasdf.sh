#!/bin/bash


for i in 01 02 04 06; do CUDA_VISIBLE_DEVICES=1 python render.py -m /data1/heungchan/output/chess/seq-${i} -s /data1/heungchan/7Scenes/chess --seq seq-${i}_novel --name novel --skip_test; done

for i in 01 02; do CUDA_VISIBLE_DEVICES=1 python render.py -m /data1/heungchan/output/fire/seq-${i} -s /data1/heungchan/7Scenes/fire --seq seq-${i}_novel --name novel --skip_test; done

for i in 02; do CUDA_VISIBLE_DEVICES=1 python render.py -m /data1/heungchan/output/heads/seq-${i} -s /data1/heungchan/7Scenes/heads --seq seq-${i}_novel --name novel --skip_test; done

for i in 01 03 04 05 08 10; do CUDA_VISIBLE_DEVICES=1 python render.py -m /data1/heungchan/output/office/seq-${i} -s /data1/heungchan/7Scenes/office --seq seq-${i}_novel --name novel --skip_test; done

for i in 02 03 06 08; do CUDA_VISIBLE_DEVICES=1 python render.py -m /data1/heungchan/output/pumpkin/seq-${i} -s /data1/heungchan/7Scenes/pumpkin --seq seq-${i}_novel --name novel --skip_test; done

for i in 01 02 05 07 08 11 13; do CUDA_VISIBLE_DEVICES=1 python render.py -m /data1/heungchan/output/redkitchen/seq-${i} -s /data1/heungchan/7Scenes/redkitchen --seq seq-${i}_novel --name novel --skip_test; done

for i in 02 03 05 06; do CUDA_VISIBLE_DEVICES=1 python render.py -m /data1/heungchan/output/stairs/seq-${i} -s /data1/heungchan/7Scenes/stairs --seq seq-${i}_novel --name novel --skip_test; done
