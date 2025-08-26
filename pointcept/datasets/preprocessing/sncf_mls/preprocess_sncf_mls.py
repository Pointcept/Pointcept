import os
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle
from tqdm import tqdm
import laspy

# dataset split
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
    return points, mean_xyz


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
            if len(tile_points) > 1000:  # skip empty/sparse tiles
                tiles.append((tile_points, (x0, y0, x1, y1)))
    return tiles


def save_tiles_and_infos(ply_file, split, output_root, tile_size=15, overlap=2.0):
    pcd = o3d.t.io.read_point_cloud(ply_file)
    points = pcd.point['positions'].numpy()

    # Try to read labels (classification) if available
    if hasattr(pcd, "point") and "scalar_Classification" in pcd.point:
        labels = np.asarray(
            pcd.point["scalar_Classification"].numpy(), dtype=np.int32)
    else:
        raise ValueError(f"{ply_file} has no classification labels!")

    # demean
    points, offset = demean_points(points)

    # split tiles
    tiles = split_into_tiles(points, tile_size, overlap)

    fname = Path(ply_file).stem
    out_dir = Path(output_root) / split / fname
    out_dir_las = Path(output_root) / split / fname / "las"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_las.mkdir(parents=True, exist_ok=True)

    infos = []
    for i, (tile_points, bounds) in enumerate(tiles):
        mask = (
            (points[:, 0] >= bounds[0]) & (points[:, 0] < bounds[2]) &
            (points[:, 1] >= bounds[1]) & (points[:, 1] < bounds[3])
        )
        tile_labels = labels[mask]

        # Shift labels from 1-8 → 0-7
        tile_labels = tile_labels - 1

        tile_path = out_dir / f"{fname}_tile_{i:05d}.npz"
        np.savez_compressed(
            tile_path,
            points=tile_points.astype(np.float32),
            labels=tile_labels.astype(np.int64)
        )

        # ---- Save LAZ (for visualization)
        laz_path = out_dir_las / f"{fname}_tile_{i:05d}.las"
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)

        # laspy requires scaled integer coords, but we can just store as float64 with scale=0.001
        las.x = tile_points[:, 0]
        las.y = tile_points[:, 1]
        las.z = tile_points[:, 2]
        las.classification = tile_labels.astype(np.uint8).reshape(-1)

        las.write(str(laz_path))

        infos.append({
            "path": str(tile_path.relative_to(output_root)),
            "origin_file": fname,
            "tile_bounds": bounds,
            "offset": offset.tolist()
        })
    return infos


def process_dataset(dataset_root, output_root, splits=SPLITS, tile_size=15, overlap=2.0):
    all_infos = {}
    for split, files in splits.items():
        infos_split = []
        print(f"Processing {split} set with {len(files)} files...")
        for ply_name in tqdm(files):
            ply_path = Path(dataset_root) / ply_name
            infos = save_tiles_and_infos(
                ply_file=ply_path,
                split=split,
                output_root=output_root,
                tile_size=tile_size,
                overlap=overlap
            )
            infos_split.extend(infos)
        # save pickle
        pkl_path = Path(output_root) / f"infos_{split}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(infos_split, f)
        all_infos[split] = infos_split
        print(f"Saved {len(infos_split)} tiles into {pkl_path}")
    return all_infos


if __name__ == "__main__":
    dataset_root = "data/sncf_mls"           # folder containing sncf_*.ply
    output_root = "data/sncf_mls/pointcept_dataset_tiles"  # where npz + pickles go

    os.makedirs(output_root, exist_ok=True)
    process_dataset(dataset_root, output_root, tile_size=15, overlap=2.0)
