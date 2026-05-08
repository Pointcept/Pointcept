"""
HK Remote Data Dataset

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import open3d as o3d
import torch

from .defaults import DefaultDataset
from .builder import DATASETS
from pointcept.utils.cache import shared_dict


@DATASETS.register_module()
class PartNetDataDataset(DefaultDataset):
    def __init__(self, if_img=True, crop_h=630, crop_w=1120, patch_size=14, **kwargs):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.patch_size = patch_size
        self.patch_h = crop_h // patch_size
        self.patch_w = crop_w // patch_size
        self.if_img = if_img
        super().__init__(**kwargs)

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        pc_data_path = os.path.join(
            data_path, "point_sample", "sample-points-all-pts-nor-rgba-10000.ply"
        )
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        pc = o3d.io.read_point_cloud(pc_data_path)
        data_dict["coord"] = np.asarray(pc.points)
        data_dict["color"] = np.asarray(pc.colors)
        data_dict["normal"] = np.asarray(pc.normals)
        data_dict["name"] = name
        data_dict["split"] = split

        if self.if_img:
            data_dict["images"] = torch.empty(
                (0, 3, self.patch_h * self.patch_size, self.patch_w * self.patch_size)
            )
            data_dict["img_num"] = np.array([0], dtype=np.int32)
            data_dict["correspondence"] = np.ones(
                (data_dict["coord"].shape[0], 0, 2), dtype=np.float32
            ) * (-1)

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
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
