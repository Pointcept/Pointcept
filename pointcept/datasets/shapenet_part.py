"""
ShapeNet Part Dataset (Unmaintained)

get processed shapenet part dataset
at "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import json
import torch
import numpy as np
from copy import deepcopy

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ShapeNetPartDataset(DefaultDataset):
    def __init__(
        self,
        if_color=False,
        split="train",
        data_root="data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super(ShapeNetPartDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.if_color = if_color
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}

        # load categories file
        self.categories = []
        self.category2part = {
            "Airplane": [0, 1, 2, 3],
            "Bag": [4, 5],
            "Cap": [6, 7],
            "Car": [8, 9, 10, 11],
            "Chair": [12, 13, 14, 15],
            "Earphone": [16, 17, 18],
            "Guitar": [19, 20, 21],
            "Knife": [22, 23],
            "Lamp": [24, 25, 26, 27],
            "Laptop": [28, 29],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Mug": [36, 37],
            "Pistol": [38, 39, 40],
            "Rocket": [41, 42, 43],
            "Skateboard": [44, 45, 46],
            "Table": [47, 48, 49],
        }
        self.token2category = {}
        with open(os.path.join(self.data_root, "synsetoffset2category.txt"), "r") as f:
            for line in f:
                ls = line.strip().split()
                self.token2category[ls[1]] = len(self.categories)
                self.categories.append(ls[0])

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # load data list
        if isinstance(self.split, str):
            self.data_list = self.load_data_list(self.split)
        elif isinstance(self.split, list):
            self.data_list = []
            for s in self.split:
                self.data_list += self.load_data_list(s)
        else:
            raise NotImplementedError

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def load_data_list(self, split):
        split_file = os.path.join(
            self.data_root,
            "train_test_split",
            "shuffled_{}_file_list.json".format(split),
        )
        if not os.path.isfile(split_file):
            raise (RuntimeError("Split file do not exist: " + split_file + "\n"))
        with open(split_file, "r") as f:
            # drop "shape_data/" and append ".txt"
            data_list = [
                os.path.join(self.data_root, data[11:] + ".txt")
                for data in json.load(f)
            ]
        return data_list

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        if data_idx in self.cache:
            coord, normal, segment, cls_token = self.cache[data_idx]
        else:
            data = np.loadtxt(self.data_list[data_idx]).astype(np.float32)
            cls_token = self.token2category[
                os.path.basename(os.path.dirname(self.data_list[data_idx]))
            ]
            coord, normal, segment = (
                data[:, :3],
                data[:, 3:6],
                data[:, 6].astype(np.int32),
            )
            self.cache[data_idx] = (coord, normal, segment, cls_token)
        name = self.get_data_name(idx)

        if self.if_color:
            color = np.zeros_like(coord)
            data_dict = dict(
                coord=coord,
                color=color,
                normal=normal,
                segment=segment,
                cls_token=cls_token,
                name=name,
            )
        else:
            data_dict = dict(coord=coord, normal=normal, cls_token=cls_token, name=name)
        return data_dict

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        basename = os.path.basename(self.data_list[data_idx]).split(".")[0]
        dirname = os.path.basename(os.path.dirname(self.data_list[data_idx]))
        return dirname + "_" + basename

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
