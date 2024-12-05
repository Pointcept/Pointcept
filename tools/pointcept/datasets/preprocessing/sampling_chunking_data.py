"""
Chunking Data

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path


def chunking_scene(
    name,
    dataset_root,
    split,
    grid_size=None,
    chunk_range=(6, 6),
    chunk_stride=(3, 3),
    chunk_minimum_size=10000,
):
    print(f"Chunking scene {name} in {split} split")
    dataset_root = Path(dataset_root)
    scene_path = dataset_root / split / name
    assets = os.listdir(scene_path)
    data_dict = dict()
    for asset in assets:
        if not asset.endswith(".npy"):
            continue
        data_dict[asset[:-4]] = np.load(scene_path / asset)
    coord = data_dict["coord"] - data_dict["coord"].min(axis=0)

    if grid_size is not None:
        grid_coord = np.floor(coord / grid_size).astype(int)
        _, idx = np.unique(grid_coord, axis=0, return_index=True)
        coord = coord[idx]
        for key in data_dict.keys():
            data_dict[key] = data_dict[key][idx]

    bev_range = coord.max(axis=0)[:2]
    x, y = np.meshgrid(
        np.arange(0, bev_range[0] + chunk_stride[0] - chunk_range[0], chunk_stride[0]),
        np.arange(0, bev_range[0] + chunk_stride[0] - chunk_range[0], chunk_stride[0]),
        indexing="ij",
    )
    chunks = np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=-1)
    chunk_idx = 0
    for chunk in chunks:
        mask = (
            (coord[:, 0] >= chunk[0])
            & (coord[:, 0] < chunk[0] + chunk_range[0])
            & (coord[:, 1] >= chunk[1])
            & (coord[:, 1] < chunk[1] + chunk_range[1])
        )
        if np.sum(mask) < chunk_minimum_size:
            continue

        chunk_data_name = f"{name}_{chunk_idx}"
        if grid_size is not None:
            chunk_split_name = (
                f"{split}_"
                f"grid{grid_size * 100:.0f}mm_"
                f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                f"stride{chunk_stride[0]}x{chunk_stride[1]}"
            )
        else:
            chunk_split_name = (
                f"{split}_"
                f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                f"stride{chunk_stride[0]}x{chunk_stride[1]}"
            )

        chunk_save_path = dataset_root / chunk_split_name / chunk_data_name
        chunk_save_path.mkdir(parents=True, exist_ok=True)
        for key in data_dict.keys():
            np.save(chunk_save_path / f"{key}.npy", data_dict[key][mask])
        chunk_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Pointcept processed ScanNet++ dataset.",
    )
    parser.add_argument(
        "--split",
        required=True,
        default="train",
        type=str,
        help="Split need to process.",
    )
    parser.add_argument(
        "--grid_size",
        default=None,
        type=float,
        help="Grid size for initial grid sampling",
    )
    parser.add_argument(
        "--chunk_range",
        default=[6, 6],
        type=int,
        nargs="+",
        help="Range of each chunk, e.g. --chunk_range 6 6",
    )
    parser.add_argument(
        "--chunk_stride",
        default=[3, 3],
        type=int,
        nargs="+",
        help="Stride of each chunk, e.g. --chunk_stride 3 3",
    )
    parser.add_argument(
        "--chunk_minimum_size",
        default=10000,
        type=int,
        help="Minimum number of points in each chunk",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )

    config = parser.parse_args()
    config.dataset_root = Path(config.dataset_root)
    data_list = os.listdir(config.dataset_root / config.split)

    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            chunking_scene,
            data_list,
            repeat(config.dataset_root),
            repeat(config.split),
            repeat(config.grid_size),
            repeat(config.chunk_range),
            repeat(config.chunk_stride),
            repeat(config.chunk_minimum_size),
        )
    )
    pool.shutdown()
