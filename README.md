
# Usage: ./train_cam.sh <dataset> <seq> <gpu_ids>
# Usage: ./train_cam.sh GreatCourt seq1 0,1,2,3

TARGET_DIR="/data1/heungchan/CambridgeLandmarks/$1"

CUDA_VISIBLE_DEVICES=$2 python convert.py -s $TARGET_DIR --resize
CUDA_VISIBLE_DEVICES=$2 python split_seq.py --input_path $TARGET_DIR/sparse/0
CUDA_VISIBLE_DEVICES=$2 python train.py -s $TARGET_DIR -m output/$1/seq1 -i images_3 --seq seq1
CUDA_VISIBLE_DEVICES=$2 python render.py -m output/$1/seq1