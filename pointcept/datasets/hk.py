"""
HK Remote Data Dataset

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import torch

from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
import numpy as np
from scipy.spatial import cKDTree


def find_closest_pair_kdtree(points):
    n = points.shape[0]
    if n < 2:
        return float("inf"), None, None

    tree = cKDTree(points)
    distances, indices = tree.query(points, k=2)
    min_dist = np.min(distances[:, 1])
    min_idx_in_distances = np.argmin(distances[:, 1])
    point1_idx = min_idx_in_distances
    point2_idx = indices[min_idx_in_distances, 1]

    return min_dist, (points[point1_idx], points[point2_idx])


@DATASETS.register_module()
class HKDataset(DefaultDataset):
    def __init__(self, crop_h=630, crop_w=1120, patch_size=14, if_img=True, **kwargs):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.patch_size = patch_size
        self.patch_h = crop_h // patch_size
        self.patch_w = crop_w // patch_size
        self.if_img = if_img
        super().__init__(**kwargs)

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
