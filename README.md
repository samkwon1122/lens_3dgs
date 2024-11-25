
# Usage: ./train_cam.sh <dataset> <seq> <gpu_ids>
# Usage: ./train_cam.sh GreatCourt seq1 0,1,2,3

TARGET_DIR="/data1/heungchan/CambridgeLandmarks/$1"

CUDA_VISIBLE_DEVICES=$2 python convert.py -s $TARGET_DIR --resize

./train_split.sh $1 $2

CUDA_VISIBLE_DEVICES=$2 python render.py -m output/$1/seq1