"""
PartNetE Datasets preprocessing

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import shutil
import argparse
import numpy as np
import trimesh
import open3d as o3d


def process_folder(target_dir):
    ply_path = os.path.join(target_dir, "pc.ply")
    label_path = os.path.join(target_dir, "label.npy")

    if os.path.exists(ply_path):
        mesh = trimesh.load(ply_path, process=False)
        coords = np.array(mesh.vertices, dtype=np.float32)
        np.save(os.path.join(target_dir, "coord.npy"), coords)

        if (
            hasattr(mesh, "vertex_normals")
            and mesh.vertex_normals is not None
            and len(mesh.vertex_normals) == len(coords)
        ):
            normals = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            pcd.estimate_normals(search_param=search_param)
            normals = np.asarray(pcd.normals).astype(np.float32)

        np.save(os.path.join(target_dir, "normal.npy"), normals)

        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
        ):
            colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
            np.save(os.path.join(target_dir, "color.npy"), colors)
        elif hasattr(mesh, "colors") and mesh.colors is not None:
            colors = np.array(mesh.colors[:, :3], dtype=np.uint8)
            np.save(os.path.join(target_dir, "color.npy"), colors)
        else:
            pass

        label_data = np.load(label_path, allow_pickle=True).item()
        segment = label_data["semantic_seg"]
        instance = label_data["instance_seg"]
        assert coords.shape[0] == segment.shape[0]
        segment_path = os.path.join(target_dir, "segment.npy")
        instance_path = os.path.join(target_dir, "instance.npy")
        np.save(segment_path, segment)
        np.save(instance_path, instance)
    else:
        print(f"Warning: pc.ply not found in {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PartNetE dataset by splitting PLY files and copying labels."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    args = parser.parse_args()
    subfolders = ["few_shot", "test"]
    total_processed_count = 0

    for subfolder in subfolders:
        current_root = os.path.join(args.dataset_root, subfolder)
        processed_in_subfolder = 0
        for dirpath, dirnames, filenames in os.walk(current_root):
            if "pc.ply" in filenames and "label.npy" in filenames:
                process_folder(dirpath)
                processed_in_subfolder += 1
