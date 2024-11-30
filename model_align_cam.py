import os

scenes = ["ShopFacade", "StMarysChurch", "GreatCourt", "KingsCollege", "OldHospital", "Street"]

for scene in scenes:
    data_path = "/data1/heungchan/CambridgeLandmarks/" + scene

    position = data_path + "/dataset_train.txt"
    new_position = data_path + "/position.txt"

    with open(position, 'r') as file:
        lines = file.readlines()[3:6]  # Skip the first 3 lines, use 3 data
        data = [line.split()[:4] for line in lines]  # Read 4 data from each line (name, x, y, z)

    with open(new_position, 'w') as file:
        for line in data:
            file.write(' '.join(line) + '\n')
    
    os.makedirs(data_path + "/sparse/00", exist_ok=True)
    
    img_undist_cmd = ("colmap" + " model_aligner \
    --input_path " + data_path + "/sparse/0 \
    --output_path " + data_path + "/sparse/00 \
    --ref_images_path " + new_position + " \
    --alignment_max_error 3.0 \
    --ref_is_gps 0")
    
    exit_code = os.system(img_undist_cmd)