"""
Preprocessing Script for GraspNet

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import argparse
from tqdm import tqdm


def split_poses(graspnet_root):
    """
    Splits the camera_poses.npy file into individual pose files for each scene.
    """
    scenes_path = os.path.join(graspnet_root, "scenes")
    scene_names = sorted(
        [
            d
            for d in os.listdir(scenes_path)
            if os.path.isdir(os.path.join(scenes_path, d))
        ]
    )

    print(f"Found {len(scene_names)} scenes. Starting preprocessing...")

    for scene_name in tqdm(scene_names, desc="Processing Scenes"):
        kinect_path = os.path.join(scenes_path, scene_name, "kinect")
        poses_file = os.path.join(kinect_path, "camera_poses.npy")
        if not os.path.exists(poses_file):
            print(f"Warning: camera_poses.npy not found in {kinect_path}, skipping.")
            continue

        all_poses = np.load(poses_file)  # Shape: (256, 4, 4)
        output_pose_dir = os.path.join(kinect_path, "pose")
        os.makedirs(output_pose_dir, exist_ok=True)

        for i in range(all_poses.shape[0]):
            pose = all_poses[i]
            output_file = os.path.join(output_pose_dir, f"{i:04d}.npy")
            np.save(output_file, pose)

    print("Preprocessing finished. Individual pose files have been created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GraspNet camera poses.")
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the root of the GraspNet dataset (the one containing the 'scenes' folder).",
    )
    args = parser.parse_args()
    split_poses(args.dataset_root)
