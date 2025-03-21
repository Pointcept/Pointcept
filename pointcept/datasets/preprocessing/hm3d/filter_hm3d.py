"""
Filtering Script for Habitat-Matterport 3D Dataset

filter out and only keep top 10,000 size of processed HM3D

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import argparse
import numpy as np
import shutil
import tqdm
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def count_scene(data_path, info_list, lock):
    data_path = Path(data_path)
    data_name = data_path.name
    data_split = data_path.parent.name
    print(f"Counting {data_name} in {data_split}..")
    data_size = np.load(data_path / "color.npy").shape[0]
    with lock:
        info_list.append(dict(name=data_name, split=data_split, size=data_size))


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Habitat-Matterport 3D dataset containing scene folders",
    )

    parser.add_argument(
        "--num_keep",
        default=10000,
        type=int,
        help="Number of scenes that kept for the dataset.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    args = parser.parse_args()

    scene_list = glob.glob(os.path.join(args.dataset_root, "*", "*"))
    if len(scene_list) <= args.num_keep:
        return
    manager = mp.Manager()
    lock = manager.Lock()
    info_list = manager.list()

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=args.num_workers)
    _ = list(
        pool.map(
            count_scene,
            scene_list,
            repeat(info_list),
            repeat(lock),
        )
    )
    name_list = np.array([scene["name"] for scene in info_list])
    split_list = np.array([scene["split"] for scene in info_list])
    size_list = np.array([scene["size"] for scene in info_list])
    remove_index = np.argsort(size_list)[: len(scene_list) - args.num_keep]

    for split in np.unique(split_list):
        os.makedirs(os.path.join(args.dataset_root, f"{split}_rm"), exist_ok=True)

    source = [
        os.path.join(args.dataset_root, split_list[i], name_list[i])
        for i in remove_index
    ]
    target = [
        os.path.join(args.dataset_root, f"{split_list[i]}_rm") for i in remove_index
    ]
    for s, t in zip(source, target):
        shutil.move(s, t)


if __name__ == "__main__":
    main_process()
