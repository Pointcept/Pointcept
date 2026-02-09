# ------------------------------------------------------------------------------
# SNCF MLS Dataset Preprocessing
#
# This script provides preprocessing utilities for the SNCF MLS dataset.
#
# Dataset License:
#   The SNCF MLS dataset is licensed under the Open Database License (ODbL).
#   You are free to use, share, and adapt the data as long as you:
#     - Attribute the source (SNCF, via the official open data platform),
#     - Share any public modifications to the database under ODbL,
#     - Keep the data open if redistributed.
#
# IMPORTANT:
#   - The dataset itself is NOT included in this repository.
#   - To access the dataset:
#       • Raw point cloud data:
#         https://ressources.data.sncf.com/explore/dataset/nuage-points-3d
#       • Annotated point cloud data (via data request form):
#         https://github.com/akharroubi/Rail3D
#
# By using this script, you agree to comply with the ODbL terms when accessing
# and using the SNCF MLS dataset.
# ------------------------------------------------------------------------------

import os
import numpy as np
import open3d as o3d
import argparse
import laspy
from pathlib import Path
from tqdm import tqdm

# Dataset split
SPLITS = {
    "train": [
        "sncf_01.ply", "sncf_02.ply", "sncf_03.ply", "sncf_05.ply",
        "sncf_06.ply", "sncf_10.ply", "sncf_12.ply", "sncf_14.ply",
        "sncf_15.ply", "sncf_16.ply",
    ],
    "val": ["sncf_09.ply", "sncf_11.ply", "sncf_13.ply"],
    "test": ["sncf_04.ply", "sncf_07.ply", "sncf_08.ply"],
}


def demean_points(points):
    mean_xyz = points[:, :3].mean(axis=0)
    points[:, :3] -= mean_xyz
    return points


def split_into_tiles(points, tile_size=15, overlap=2.0):
    min_bound = points[:, :2].min(axis=0)
    max_bound = points[:, :2].max(axis=0)

    tiles = []
    x_range = np.arange(min_bound[0], max_bound[0], tile_size - overlap)
    y_range = np.arange(min_bound[1], max_bound[1], tile_size - overlap)

    for x0 in x_range:
        for y0 in y_range:
            x1, y1 = x0 + tile_size, y0 + tile_size
            mask = (
                (points[:, 0] >= x0) & (points[:, 0] < x1) &
                (points[:, 1] >= y0) & (points[:, 1] < y1)
            )
            tile_points = points[mask]
            if len(tile_points) > 1000:  # Skip empty/sparse tiles
                tiles.append((tile_points, (x0, y0, x1, y1)))
    return tiles


def save_tiles_and_infos(ply_file, split, output_root, tile_size=15, overlap=2.0):
    pcd = o3d.t.io.read_point_cloud(ply_file)
    points = pcd.point['positions'].numpy()

    # Extract labels
    if hasattr(pcd, "point") and "scalar_Classification" in pcd.point:
        labels = np.asarray(
            pcd.point["scalar_Classification"].numpy(), dtype=np.int32)
    else:
        raise ValueError(f"{ply_file} has no classification labels!")

    # Extract scalar_Intensity
    if hasattr(pcd, "point") and "scalar_Intensity" in pcd.point:
        intensities = np.asarray(
            pcd.point["scalar_Intensity"].numpy(), dtype=np.uint8)
        intensities = intensities.reshape(-1, 1)
    else:
        intensities = np.zeros((points.shape[0], 1), dtype=np.uint8)

    # Demean
    points = demean_points(points)

    # Split tiles
    tiles = split_into_tiles(points, tile_size, overlap)

    fname = Path(ply_file).stem
    out_dir = Path(output_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (tile_points, bounds) in enumerate(tiles):
        mask = (
            (points[:, 0] >= bounds[0]) & (points[:, 0] < bounds[2]) &
            (points[:, 1] >= bounds[1]) & (points[:, 1] < bounds[3])
        )
        tile_labels = labels[mask]
        tile_intensities = intensities[mask]

        # Shift labels from 1-8 to 0-7
        tile_labels = tile_labels - 1

        # Save data
        tile_path = out_dir / f"{fname}_tile_{i:05d}"

        data_dict = dict(
            coord=tile_points.astype(np.float32),
            segment=tile_labels.astype(np.int16),
            strength=tile_intensities.astype(np.uint8)
        )
        os.makedirs(tile_path, exist_ok=True)

        for key, arr in data_dict.items():
            np.save(os.path.join(tile_path, f"{key}.npy"), arr)

        # Save LAS (for visualization)
        laz_path = tile_path / f"{fname}_tile_{i:05d}.las"
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)

        las.x = tile_points[:, 0]
        las.y = tile_points[:, 1]
        las.z = tile_points[:, 2]
        las.intensity = tile_intensities.astype(np.uint8).reshape(-1)
        las.classification = tile_labels.astype(np.uint8).reshape(-1)

        las.write(str(laz_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located.",
    )
    parser.add_argument(
        "--tile_size",
        type=float,
        default=30.0,
        help="Size of each tile (in meters). Default: 30.0",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=5.0,
        help="Overlap between tiles (in meters). Default: 5.0",
    )

    config = parser.parse_args()

    # Create output directories and process each split
    for split, files in SPLITS.items():
        print(f"Processing {split} set with {len(files)} files...")
        os.makedirs(os.path.join(config.output_root, split), exist_ok=True)
        for ply_name in tqdm(files):
            ply_path = Path(config.dataset_root) / ply_name
            save_tiles_and_infos(
                ply_file=ply_path,
                split=split,
                output_root=config.output_root,
                tile_size=config.tile_size,
                overlap=config.overlap
            )
