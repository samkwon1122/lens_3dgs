import struct
import numpy as np
import argparse
import collections
import struct
import os
import shutil

# dataset = "/data1/heungchan/CambridgeLandmarks/KingsCollege/dataset_test.txt"

# with open(dataset, 'r') as file:
#     lines = file.readlines()

# with open(dataset, 'w') as file:
#     for i, line in enumerate(lines):
#         if i < 3:
#             file.write(line)
#         else:
#             file.write("input/" + line)

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "camera_center"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            camera_center = -qvec2rotmat(qvec).T @ tvec
            
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids, camera_center=camera_center)
    return images

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
args = parser.parse_args()

data_path = "/data1/heungchan/7Scenes/"
scene = args.scene

if "GreatCourt" in scene:
    seq_list = ["seq2", "seq3", "seq5"]
elif "ShopFacade" in scene:
    seq_list = ["seq2"]
elif "KingsCollege" in scene:
    seq_list = ["seq1", "seq4", "seq5", "seq6", "seq8"]
elif "OldHospital" in scene:
    seq_list = ["seq1", "seq2", "seq3", "seq5", "seq6", "seq7", "seq9"]
elif "StMarysChurch" in scene:
    seq_list = ["seq1", "seq2", "seq4", "seq6", "seq7", "seq8", "seq9", "seq10", "seq11", "seq12", "seq14"]
elif "Street" in scene:
    seq_list = ["img_east", "img_north", "img_south", "img_west"]
    
elif "chess" in scene:
    seq_list = ["seq-01", "seq-02", "seq-04", "seq-06"]
elif "fire" in scene:
    seq_list = ["seq-01", "seq-02"]
elif "heads" in scene:
    seq_list = ["seq-02"]
elif "office" in scene:
    seq_list = ["seq-01", "seq-03", "seq-04", "seq-05", "seq-08", "seq-10"]
elif "pumpkin" in scene:
    seq_list = ["seq-02", "seq-03", "seq-06", "seq-08"]
elif "redkitchen" in scene:
    seq_list = ["seq-01", "seq-02", "seq-05", "seq-07", "seq-08", "seq-11", "seq-13"]
elif "stairs" in scene:
    seq_list = ["seq-02", "seq-03", "seq-05", "seq-06"]

train_file = data_path + scene + "/dataset_train_colmap.txt"
test_file = data_path + scene + "/dataset_test_colmap.txt"

with open(train_file, 'w') as file:
    with open(test_file, 'w') as file2:
        file.write("LENS with 3DGS\n")
        file.write("ImageFile, Camera Position [X Y Z W P Q R]\n\n")
        file2.write("LENS with 3DGS\n")
        file2.write("ImageFile, Camera Position [X Y Z W P Q R]\n\n")
        
        images = read_images_binary(os.path.join(data_path, scene, "sparse/00/images.bin"))
        
        for image_id, image in images.items():
            if image.name.split("/")[0] in seq_list:
                file.write(f"images_2/{image.name} {image.camera_center[0]} {image.camera_center[1]} {image.camera_center[2]} {image.qvec[0]} {image.qvec[1]} {image.qvec[2]} {image.qvec[3]}\n")
            else:
                file2.write(f"images_2/{image.name} {image.camera_center[0]} {image.camera_center[1]} {image.camera_center[2]} {image.qvec[0]} {image.qvec[1]} {image.qvec[2]} {image.qvec[3]}\n") 

novel_file = os.path.join(data_path, scene, f"dataset_novel.txt")
shutil.copyfile(train_file, novel_file)

if args.scene == "Street":
    with open(novel_file, 'a') as file:
        for seq in seq_list:
            images = read_images_binary(os.path.join(data_path, scene, f"sparse/{seq}_novel/images.bin"))
                
            for image_id, image in images.items():
                file.write(f"novel/{seq}_novel/{image.name} {image.camera_center[0]} {image.camera_center[1]} {image.camera_center[2]} {image.qvec[0]} {image.qvec[1]} {image.qvec[2]} {image.qvec[3]}\n")
    
    exit(0)

with open(novel_file, 'a') as file:
    for seq in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]:
        seq_path = os.path.join(data_path, scene, f"sparse/seq-{seq}_novel/images.bin")
        if os.path.exists(seq_path):
            images = read_images_binary(seq_path)
                
            for image_id, image in images.items():
                file.write(f"novel/seq-{seq}_novel/{image.name} {image.camera_center[0]} {image.camera_center[1]} {image.camera_center[2]} {image.qvec[0]} {image.qvec[1]} {image.qvec[2]} {image.qvec[3]}\n")
                