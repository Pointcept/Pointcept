"""
Preprocessing Script for HK Remote

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import multiprocessing as mp


def handle_process(obj_file_paths, output_directory):
    num_points_to_sample = 10000000
    for obj_file_path in obj_file_paths:
        try:
            print(f"Loading mesh from: {obj_file_path}")
            parent_name = obj_file_path.parent.name
            save_path = output_directory / parent_name
            if save_path.exists():
                print(f"exist {str(parent_name)}")
                continue
            mesh = o3d.io.read_triangle_mesh(obj_file_path, enable_post_processing=True)

            if not mesh.has_vertices():
                print("Error: Mesh could not be loaded.")
                exit()

            if not mesh.has_textures():
                print(
                    "Warning: Mesh does not have textures. Sampled colors will be black or default."
                )

            mesh.compute_vertex_normals()
            print("Mesh loaded successfully. Proceeding to sample points...")

            pcd = mesh.sample_points_uniformly(number_of_points=num_points_to_sample)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            if colors.shape[0] == 0:
                colors = np.zeros_like(points)
            normals = np.asarray(pcd.normals)
            if normals.shape[0] == 0:
                normals = np.zeros_like(points)

            print(f"\nSuccessfully sampled {len(points)} points.")
            print(f"Points array shape: {points.shape}")  # (N, 3) for XYZ coordinates
            print(
                f"Colors array shape: {colors.shape}"
            )  # (N, 3) for RGB values (0-1 range)
            print(f"Normals array shape: {normals.shape}")  # (N, 3) for normal vectors

            output_coord_path = output_directory / parent_name / "coord.npy"
            output_color_path = output_directory / parent_name / "color.npy"
            output_normal_path = output_directory / parent_name / "normal.npy"
            (output_directory / parent_name).mkdir(exist_ok=True)

            np.save(output_coord_path, points)
            np.save(output_color_path, (colors * 255).astype(np.int32))
            np.save(output_normal_path, normals)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            print(
                f"\nSampled point cloud saved to: {str(output_directory/parent_name)}"
            )
        except:
            print(f"fail {parent_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--thread_id",
        default=0,
        type=int,
        help="Thread id for parallel processing",
    )
    config = parser.parse_args()
    output_directory = Path(config.output_root) / "train"
    root_directory = Path(config.dataset_root)
    output_directory.mkdir(exist_ok=True)
    obj_file_paths_generator = root_directory.rglob("*.obj")
    obj_file_paths = list(obj_file_paths_generator)
    print(f"Found {len(obj_file_paths)} .obj files.")

    obj_file_paths_list = np.array_split(obj_file_paths, config.num_workers)
    obj_file_paths_ = obj_file_paths_list[config.thread_id]

    handle_process(obj_file_paths_, output_directory)
