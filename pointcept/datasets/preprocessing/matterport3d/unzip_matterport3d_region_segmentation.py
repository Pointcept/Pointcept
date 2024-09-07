"""
Preprocessing Script for Matterport3D (Unzipping)
modified from official preprocess code.

Author: Chongjie Ye (chongjieye@link.cuhk.edu.cn)
Modified by: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import argparse
import os
import zipfile
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def unzip_file(input_path, output_path):
    print(f"Unzipping {input_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as zip_ref:
        zip_ref.extractall(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unzip all "region_segmentations.zip" files in a directory'
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Path to input directory containing ZIP files",
        required=True,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Path to output directory for extracted files",
        default=None,
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = args.dataset_root
    args.output_root = os.path.join(args.output_root, "v1", "scans")

    file_list = glob.glob(
        os.path.join(args.dataset_root, "v1", "scans", "*", "region_segmentations.zip")
    )

    # Preprocess data.
    print("Unzipping region_segmentations.zip in Matterport3D...")
    pool = ProcessPoolExecutor(max_workers=args.num_workers)
    _ = list(
        pool.map(
            unzip_file,
            file_list,
            repeat(args.output_root),
        )
    )
