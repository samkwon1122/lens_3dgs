import struct
import numpy as np
from scipy.spatial import KDTree
import argparse
import collections
import struct
import os
import shutil

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "camera_center"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

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

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

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

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

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

def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")

def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D

def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    args = parser.parse_args()
    
    images = read_images_binary(os.path.join(args.input_path, "images.bin"))
    points3D = read_points3D_binary(os.path.join(args.input_path, "points3D.bin"))

    d_scene = 0.05 # Scene과의 최소 거리
    d_view = 0.1   # Train 카메라와의 최대 거리
    resolution = 0.05  # Candidate 위치 생성 간격
    theta = 15
    
    base_dir, last_dir = os.path.split(args.input_path.rstrip('/'))
    novel_dir = os.path.join(base_dir, last_dir + "_novel")
    os.makedirs(novel_dir, exist_ok=True)
    output_path = os.path.join(novel_dir, "images.bin")
    
    camera_path = os.path.join(args.input_path, "cameras.bin")
    shutil.copy(camera_path, os.path.join(novel_dir, "cameras.bin"))
    
    points3D_path = os.path.join(args.input_path, "points3D.bin")
    shutil.copy(points3D_path, os.path.join(novel_dir, "points3D.bin"))
    
    # 1. original camera centers
    camera_centers = []
    for image in images.values():
        camera_center = -qvec2rotmat(image.qvec).T @ image.tvec
        camera_centers.append(camera_center)    
    camera_centers = np.array(camera_centers)
    
    # 2. bounding box
    min_corner = np.min(camera_centers, axis=0)
    max_corner = np.max(camera_centers, axis=0)
    lengths = max_corner - min_corner
    
    d_view = min(lengths) / 2

    num_cubes = 1000
    cube_volume = np.prod(lengths) / num_cubes
    cube_side_length = cube_volume ** (1/3)
    
    # 3. candidates    
    candidates = []
    
    x_range = np.arange(min_corner[0], max_corner[0], cube_side_length)
    y_range = np.arange(min_corner[1], max_corner[1], cube_side_length)
    z_range = np.arange(min_corner[2], max_corner[2], cube_side_length)
    
    candidates = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
    
    # 4. filtering
    scene_points = np.array([point.xyz for point in points3D.values()])
    scene_kdtree = KDTree(scene_points)
    train_kdtree = KDTree(camera_centers)
    
    dist_scene, _ = scene_kdtree.query(candidates)
    dist_train, nearest_train_idx = train_kdtree.query(candidates)
    valid_mask = (dist_scene > d_scene) & (dist_train < d_view)
    
    valid_candidates = candidates[valid_mask]
    nearest_indices = nearest_train_idx[valid_mask]
    
    # 5. novel views
    novel_views = {}
    
    for i, (center, train_idx) in enumerate(zip(valid_candidates, nearest_indices)):
        train_image = list(images.values())[train_idx]
        random_angles = np.radians(np.random.uniform(-theta / 2, theta / 2, 3))
        random_rotation = np.array([
            [np.cos(random_angles[2]) * np.cos(random_angles[1]), 
             np.cos(random_angles[2]) * np.sin(random_angles[1]) * np.sin(random_angles[0]) - np.sin(random_angles[2]) * np.cos(random_angles[0]), 
             np.cos(random_angles[2]) * np.sin(random_angles[1]) * np.cos(random_angles[0]) + np.sin(random_angles[2]) * np.sin(random_angles[0])],
            [np.sin(random_angles[2]) * np.cos(random_angles[1]), 
             np.sin(random_angles[2]) * np.sin(random_angles[1]) * np.sin(random_angles[0]) + np.cos(random_angles[2]) * np.cos(random_angles[0]), 
             np.sin(random_angles[2]) * np.sin(random_angles[1]) * np.cos(random_angles[0]) - np.cos(random_angles[2]) * np.sin(random_angles[0])],
            [-np.sin(random_angles[1]), 
             np.cos(random_angles[1]) * np.sin(random_angles[0]), 
             np.cos(random_angles[1]) * np.cos(random_angles[0])]
        ])
        new_qvec = rotmat2qvec(random_rotation @ qvec2rotmat(train_image.qvec))
        
        novel_views[i] = Image(
            id=i,  # 새로운 ID
            qvec=new_qvec,  # 새로운 회전
            tvec=-qvec2rotmat(new_qvec) @ center,  # 새로운 위치
            camera_id=train_image.camera_id,  # 동일 카메라 ID
            name=f"novel_view_{i}.png",  # 새 이미지 이름
            xys=np.empty((0, 2)),  # Empty 2D points
            point3D_ids=np.empty(0, dtype=int),
            camera_center=center
        )

    # 6. save novel views
    write_images_binary(novel_views, output_path)
    print(f"{len(novel_views)} novel views saved to {output_path}")

if __name__ == "__main__":
    main()