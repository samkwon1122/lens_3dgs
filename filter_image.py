# 이상한 이미지 거르기
# 사용하지 않음. 이상한 이미지도 그냥 받아들여.

import numpy as np
import os
from PIL import Image

data = "/data1/heungchan/CambridgeLandmarks/KingsCollege"
seq = data + "/images_3/seq1"
novel = "/data1/heungchan/output/KingsCollege/seq1/novel/ours_30000/renders"

image_files = [f for f in os.listdir(seq) if f.endswith('.png')]

def calculate_mean_rgb(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return np.mean(image_array, axis=(0, 1))

total_mean_rgb = np.zeros(3)
for image_file in image_files:
    image_path = os.path.join(seq, image_file)
    total_mean_rgb += calculate_mean_rgb(image_path)

total_mean_rgb /= len(image_files)
print("Total mean of images (RGB):", total_mean_rgb)

image_files = [f for f in os.listdir(novel) if f.endswith('.png')]

threshold = 50  # Define a threshold for deviation

for image_file in image_files:
    image_path = os.path.join(novel, image_file)
    mean_rgb = calculate_mean_rgb(image_path)
    if np.any(np.abs(mean_rgb - total_mean_rgb) > threshold):
        print(image_file, total_mean_rgb - mean_rgb)