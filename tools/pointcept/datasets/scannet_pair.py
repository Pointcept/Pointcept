"""
ScanNet Pair Dataset (Frame-level contrastive view)

Refer PointContrast

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class ScanNetPairDataset(Dataset):
    def __init__(
        self,
        data_root="data/scannet_pair",
        overlap_threshold=0.3,
        view1_transform=None,
        view2_transform=None,
        loop=1,
        **kwargs
    ):
        super(ScanNetPairDataset, self).__init__()
        self.data_root = data_root
        self.overlap_threshold = overlap_threshold
        self.view1_transform = Compose(view1_transform)
        self.view2_transform = Compose(view2_transform)
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info("Totally {} x {} samples.".format(len(self.data_list), self.loop))

    def get_data_list(self):
        data_list = []
        overlap_list = glob.glob(
            os.path.join(self.data_root, "*", "pcd", "overlap.txt")
        )
        for overlap_file in overlap_list:
            with open(overlap_file) as f:
                overlap = f.readlines()
            overlap = [pair.strip().split() for pair in overlap]
            data_list.extend(
                [
                    pair[:2]
                    for pair in overlap
                    if float(pair[2]) > self.overlap_threshold
                ]
            )
        return data_list

    def get_data(self, idx):
        pair = self.data_list[idx % len(self.data_list)]
        view1_dict = torch.load(self.data_root + pair[0])
        view2_dict = torch.load(self.data_root + pair[1])
        return view1_dict, view2_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        view1_dict, view2_dict = self.get_data(idx)
        view1_dict = self.view1_transform(view1_dict)
        view2_dict = self.view2_transform(view2_dict)
        data_dict = dict()
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict

    def prepare_test_data(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
