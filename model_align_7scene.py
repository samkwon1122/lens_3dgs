import os
import numpy as np

scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

def get_position(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        Rt = np.array([list(map(float, line.split())) for line in lines[:3]])
        R = Rt[:, :3]
        t = Rt[:, 3]
        camera_center = -np.linalg.inv(R).dot(t)
        return camera_center

for scene in scenes:
    data_path = "/data1/heungchan/7Scenes/" + scene

    position1 = data_path + "/seq-01/frame-000000.pose.txt"
    position2 = data_path + "/seq-01/frame-000001.pose.txt"
    position3 = data_path + "/seq-01/frame-000002.pose.txt"
    new_position = data_path + "/position.txt"

    data1 = list(map(str, get_position(position1)))
    data2 = list(map(str, get_position(position2)))
    data3 = list(map(str, get_position(position3)))
    
    with open(new_position, 'w') as file:
        file.write("seq-01/frame-000000.color.png " + ' '.join(data1) + '\n')
        file.write("seq-01/frame-000001.color.png " + ' '.join(data2) + '\n')
        file.write("seq-01/frame-000002.color.png " + ' '.join(data3) + '\n')
    
    os.makedirs(data_path + "/sparse/00", exist_ok=True)
    
    img_undist_cmd = ("colmap" + " model_aligner \
    --input_path " + data_path + "/sparse/0 \
    --output_path " + data_path + "/sparse/00 \
    --ref_images_path " + new_position + " \
    --alignment_max_error 3.0 \
    --ref_is_gps 0")
    
    exit_code = os.system(img_undist_cmd)