"""
PartNetE Datasets

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
from collections.abc import Sequence
import numpy as np
import json

from .defaults import DefaultDataset
from .builder import DATASETS

from pointcept.utils.cache import shared_dict


@DATASETS.register_module()
class PartNetEDataset(DefaultDataset):
    def __init__(
        self, class_names, num_parts, data_root, meta_path, split, *args, **kwargs
    ):
        self.categories = class_names
        self.num_parts = num_parts
        self.num_part_offset = np.concatenate(([0], np.cumsum(self.num_parts)))
        with open(meta_path, "r", encoding="utf-8") as file:
            meta_data = json.load(file)
        self.category2part = {}
        self.parts = []
        for class_id, class_name in enumerate(self.categories):
            self.category2part[class_name] = self.num_part_offset[class_id] + list(
                range(self.num_parts[class_id])
            )
            category_part_name = [
                class_name + "_" + part_name
                for part_name in ["other"] + meta_data[class_name]
            ]
            self.parts.extend(category_part_name)
        super().__init__(data_root=data_root, split=split, *args, **kwargs)

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*/*"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*/*"))
        else:
            raise NotImplementedError
        return data_list

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, id_name = os.path.split(file_path)
        object_name = os.path.basename(dir_path)
        data_name = f"{object_name}_{id_name}"
        return data_name

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split

        object_name = name.split("_")[0]
        cls_token = self.categories.index(object_name)
        data_dict["cls_token"] = cls_token

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = (
                data_dict["segment"].reshape([-1]).astype(np.int32)
                + self.num_part_offset[cls_token]
                + 1
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        return data_dict
