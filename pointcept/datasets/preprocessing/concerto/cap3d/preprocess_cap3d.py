"""
Preprocessing Script for Cap3D

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import torch
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from pathlib import Path
import multiprocessing as mp
import zipfile
import glob
import json
import math

NUM_VIEWS = 4
MAX_DEPTH = 5.0


def recover_depth_from_image(depth_image_path, max_depth):
    with Image.open(depth_image_path) as depth_pil:
        depth_uint16 = np.array(depth_pil)
    invalid_depth_mask = depth_uint16 == 65535
    depth_normalized_float = depth_uint16.astype(np.float32) / 65535.0
    depth_meters = depth_normalized_float * max_depth
    return depth_meters, invalid_depth_mask


def normalize_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    bbox_min = np.min(point_cloud, axis=0)
    bbox_max = np.max(point_cloud, axis=0)
    bbox_size = bbox_max - bbox_min
    max_dim = np.max(bbox_size)
    if max_dim < 1e-8:
        return point_cloud - bbox_min

    scale = 1.0 / max_dim
    scaled_point_cloud = point_cloud * scale
    new_bbox_min = np.min(scaled_point_cloud, axis=0)
    new_bbox_max = np.max(scaled_point_cloud, axis=0)
    center = (new_bbox_min + new_bbox_max) / 2.0
    offset = -center
    normalized_point_cloud = scaled_point_cloud + offset

    return normalized_point_cloud


def get_cam_params(cam_params_file, width, height):
    with open(cam_params_file, "r") as f:
        data = json.load(f)

    x_vector = data["x"]
    y_vector = data["y"]
    z_vector = data["z"]
    origin = data["origin"]

    rotation_matrix = np.array([x_vector, y_vector, z_vector]).T

    translation_vector = np.array(origin)

    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector

    x_fov = data["x_fov"]
    y_fov = data["y_fov"]

    cx = width / 2.0
    cy = height / 2.0

    fx = (width / 2.0) / math.tan(x_fov / 2.0)
    fy = (height / 2.0) / math.tan(y_fov / 2.0)

    s = 0

    K = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])

    return rt_matrix, K


def correspondenceGet(depth, invalid_depth_mask, T, K, width, height):
    pixel = np.transpose(np.indices((width, height)), (2, 1, 0))
    pixel = pixel.reshape((-1, 2))
    pixel = np.hstack((pixel, np.ones((pixel.shape[0], 1))))
    depth = depth.reshape((-1, 1))
    invalid_depth_mask = invalid_depth_mask.reshape((-1, 1))
    valid = ~np.logical_or(np.isinf(depth), invalid_depth_mask).squeeze(-1)
    coord = np.zeros_like(pixel, dtype=np.float32)
    coord[valid] = depth[valid] * (np.linalg.inv(K) @ pixel[valid].T).T  # coord_camera
    coord[valid] = coord[valid] @ T[:3, :3].T + T[:3, 3]  # column then row

    pixel = pixel[valid]
    coord = coord[valid]
    if coord.shape[0] == 0:
        return None
    pixel = pixel[:, :2]
    coord_dict = {"pixel": pixel, "coord": coord}
    return coord_dict


def correspondenceSave(
    depth_paths, data_name, point_cloud_gt, cam_params_paths, output_root
):
    """
    Calculates and saves the correspondences for all views of a given mesh and point cloud.
    """
    # Build a KDTree for the ground truth point cloud for fast querying.
    tree = cKDTree(point_cloud_gt)

    # Iterate over each predefined view.
    for view_idx in range(len(depth_paths)):
        # Construct file paths.
        cam_params_path = cam_params_paths[view_idx]
        depth_path = depth_paths[view_idx]

        view_name = cam_params_path.split("/")[-1][:-5]

        # Load camera parameters and image dimensions.
        depth, invalid_depth_mask = recover_depth_from_image(depth_path, MAX_DEPTH)
        height, width = depth.shape
        T, K = get_cam_params(cam_params_path, width, height)

        coord_dict = correspondenceGet(depth, invalid_depth_mask, T, K, width, height)
        if coord_dict is None:
            correspondences = -np.ones((1, 3))
        else:
            pixels_ = coord_dict["pixel"]
            coords_ = coord_dict["coord"]
            dis, idx = tree.query(coords_, k=1)
            idx_valid = idx[dis < 0.01]
            pixels_valid = pixels_[dis < 0.01]
            correspondences = np.hstack((pixels_valid, idx_valid.reshape(-1, 1)))
        output_dir = output_root / data_name
        os.makedirs(output_dir, exist_ok=True)
        output_filename = output_dir / f"{view_name}.npy"
        np.save(output_filename, correspondences)


def handle_process(pt_path, output_root, cam_root):
    """
    The complete processing pipeline for a single GLB file.
    """
    data_name = pt_path.stem

    if not pt_path.exists():
        return
    cam_path = cam_root / f"{data_name}"
    cam_path_zip = cam_root / f"{data_name}.zip"

    # Check if files exist, just in case.
    if not cam_path.exists():
        if not cam_path_zip.exists():
            print(f"not exist {str(cam_path)} and its zip file")
            return
        with zipfile.ZipFile(cam_path_zip, "r") as zip_ref:
            zip_ref.extractall(cam_path)

    # Load the point cloud.
    point_cloud = torch.load(pt_path)
    point_cloud_np = (
        point_cloud.cpu().numpy()
        if hasattr(point_cloud, "numpy")
        else np.array(point_cloud)
    )
    point_cloud_np = point_cloud_np[:3, :].T
    depth_paths = glob.glob(os.path.join(cam_path, "*_depth.png"))
    if len(depth_paths) >= NUM_VIEWS:
        depth_paths = sorted(depth_paths)[:: len(depth_paths) // NUM_VIEWS]
    else:
        depth_paths = sorted(depth_paths)
    cam_params_paths = glob.glob(os.path.join(cam_path, "*.json"))
    if os.path.isfile(os.path.join(cam_path, "transforms_train.json")):
        cam_params_paths.remove(os.path.join(cam_path, "transforms_train.json"))
    if os.path.isfile(os.path.join(cam_path, "info.json")):
        cam_params_paths.remove(os.path.join(cam_path, "info.json"))
    if len(depth_paths) >= NUM_VIEWS:
        cam_params_paths = sorted(cam_params_paths)[
            :: len(cam_params_paths) // NUM_VIEWS
        ]
    else:
        cam_params_paths = sorted(cam_params_paths)

    valid_depth_paths = {
        os.path.basename(p).replace("_depth.png", ""): p for p in depth_paths
    }
    valid_cam_params_paths = {
        os.path.basename(p).replace(".json", ""): p for p in cam_params_paths
    }
    common_ids = set(valid_depth_paths.keys()) & set(valid_cam_params_paths.keys())
    sorted_common_ids = sorted(list(common_ids))

    valid_depth_paths_ = [valid_depth_paths[uid] for uid in sorted_common_ids]
    valid_cam_params_paths_ = [valid_cam_params_paths[uid] for uid in sorted_common_ids]
    # Calculate and save correspondences for all views.
    correspondenceSave(
        valid_depth_paths_,
        data_name,
        point_cloud_np,
        valid_cam_params_paths_,
        output_root,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2D-3D correspondences for GLB models and point clouds."
    )
    parser.add_argument(
        "--cam_root",
        type=Path,
        required=True,
        help="Root directory containing camera parameters.",
    )
    parser.add_argument(
        "--point_cloud_root",
        type=Path,
        required=True,
        help="Directory containing .pt point cloud files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Output directory to save correspondence files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of processes to use for processing.",
    )
    parser.add_argument(
        "--thread_id", type=int, default=0, help="thread id to use for processing."
    )
    config = parser.parse_args()

    # Ensure the output directory exists.
    os.makedirs(config.output_root, exist_ok=True)

    print("Scanning for .pt files...")
    pt_files = list(config.point_cloud_root.glob("**/*.pt"))
    print(f"Found {len(pt_files)} .pt files.")

    # Load scene paths
    pt_files_list = np.array_split(pt_files, config.num_workers)
    pt_files_ = pt_files_list[config.thread_id]
    # Preprocess data.
    print("Processing scenes...")
    for pt_files_i in pt_files_:
        handle_process(
            pt_files_i,
            config.output_root,
            config.cam_root,
        )

    print("\nAll files processed.")
