import numpy as np
import open3d as o3d

def filter_points_by_grid(ply_file, output_file, grid_size, min_points_per_voxel):
    # PLY 파일 로드
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    # 점군의 범위 계산
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # 격자 크기에 따라 공간 분할
    grid_indices = np.floor((points - min_bound) / grid_size).astype(int)

    # 각 격자에 있는 포인트 계산
    unique_indices, counts = np.unique(grid_indices, axis=0, return_counts=True)
    valid_voxels = unique_indices[counts >= min_points_per_voxel]

    # 유효한 격자에 해당하는 포인트 필터링
    valid_mask = np.isin(grid_indices, valid_voxels).all(axis=1)
    filtered_points = points[valid_mask]

    # 필터링된 점군 생성 및 저장
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    o3d.io.write_point_cloud(output_file, filtered_pcd)

# 예시 실행
input_ply_file = "/home/vision/heungchan/gaussian-splatting/output/ShopFacade/seq2/point_cloud/iteration_30000/point_cloud.ply"  # 입력 PLY 파일 경로
output_ply_file = "/home/vision/heungchan/gaussian-splatting/output/ShopFacade/seq2/point_cloud/iteration_30000/output.ply"  # 출력 PLY 파일 경로
grid_size = 0.1  # 격자의 크기
min_points_per_voxel = 300  # 격자당 최소 포인트 수

filter_points_by_grid(input_ply_file, output_ply_file, grid_size, min_points_per_voxel)
