"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import torch
import numpy as np
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to the ScanNet++ dataset containing data/metadata/splits.",
    )
    parser.add_argument(
        "--pth_root",
        required=True,
        help="Path to the ScanNet++ official toolkit generated pth files.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located.",
    )
    parser.add_argument(
        "--ignore_index",
        default=-1,
        type=int,
        help="Default ignore index.",
    )
    config = parser.parse_args()

    print("Loading meta data...")
    config.data_root = Path(config.data_root)
    config.output_root = Path(config.output_root)

    val_list = np.loadtxt(
        config.data_root / "splits" / "nvs_sem_val.txt",
        dtype=str,
    )
    print("Num samples in validation split:", len(val_list))

    test_list = np.loadtxt(
        config.data_root / "splits" / "sem_test.txt",
        dtype=str,
    )
    print("Num samples in testing split:", len(test_list))
    val_output = Path(config.output_root) / "val_vtx"
    test_output = Path(config.output_root) / "test_vtx"

    sp_val_ath = Path(config.output_root) / "val"

    val_vtx_path = Path(config.pth_root) / "val_vtx"
    test_vtx_path = Path(config.pth_root) / "test_vtx"
    for i, scene_id in enumerate(val_list):
        data = torch.load(val_vtx_path / f"{scene_id}.pth")
        coord = data["vtx_coords"].astype(np.float32)
        color = (data["vtx_colors"] * 255).astype(np.uint8)
        normal = data["vtx_normals"].astype(np.float32)
        superpoint = np.load(sp_val_ath / scene_id / "superpoint.npy")

        # superpoint = data['vtx_superpoints']

        segment = data["vtx_labels"]
        instance = data["vtx_instance_labels"]

        save_path = val_output / scene_id
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "coord.npy", coord)
        np.save(save_path / "color.npy", color)
        np.save(save_path / "normal.npy", normal)
        np.save(save_path / "superpoint.npy", superpoint)
        np.save(save_path / "segment.npy", segment)
        np.save(save_path / "instance.npy", instance)

    for i, scene_id in enumerate(test_list):
        data = torch.load(test_vtx_path / f"{scene_id}.pth")
        coord = data["vtx_coords"].astype(np.float32)
        color = (data["vtx_colors"] * 255).astype(np.uint8)
        normal = data["vtx_normals"].astype(np.float32)
        superpoint = data["vtx_superpoints"]

        save_path = test_output / scene_id
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "coord.npy", coord)
        np.save(save_path / "color.npy", color)
        np.save(save_path / "normal.npy", normal)
        np.save(save_path / "superpoint.npy", superpoint)
