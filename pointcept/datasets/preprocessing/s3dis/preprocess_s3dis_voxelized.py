"""
Preprocessing Script for S3DIS
Parsing normal vectors has a large consumption of memory. Please reduce max_workers if memory is limited.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import torch
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from pointcept.datasets.transform import GridSample


def voxelize_parser(data_path, dataset_root, output_root, voxel_size):
    print(f"Parsing data: {data_path}")
    out_path = data_path.replace(dataset_root, output_root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = torch.load(data_path)
    data = GridSample(
        grid_size=voxel_size, hash_type="fnv", mode="train", keys=data.keys()
    )(data)
    torch.save(data, out_path)


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to processed S3DIS dataset"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where area folders will be located",
    )
    parser.add_argument(
        "--voxel_size", default=0.01, type=float, help="Voxel size for voxelization"
    )
    args = parser.parse_args()

    data_list = glob.glob(os.path.join(args.dataset_root, "*/*.pth"))
    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    # pool = ProcessPoolExecutor(max_workers=1)
    _ = list(
        pool.map(
            voxelize_parser,
            data_list,
            repeat(args.dataset_root),
            repeat(args.output_root),
            repeat(args.voxel_size),
        )
    )


if __name__ == "__main__":
    main_process()
