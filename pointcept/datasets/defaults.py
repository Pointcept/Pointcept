"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import json

import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from torchvision.transforms import InterpolationMode
from PIL import Image
from torchvision.transforms import transforms as T
import torch.nn.functional as F

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS

INTERPOLATION_MODE = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


@DATASETS.register_module()
class DefaultDataset(Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
    ]

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop, os.path.basename(self.data_root), split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            if os.path.isfile(os.path.join(self.data_root, split)):
                with open(os.path.join(self.data_root, split)) as f:
                    data_list += [
                        os.path.join(self.data_root, data) for data in json.load(f)
                    ]
            else:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        return data_list

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

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def get_split_name(self, idx):
        return os.path.basename(
            os.path.dirname(self.data_list[idx % len(self.data_list)])
        )

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class DefaultImagePointDataset(Dataset):
    PC_VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
    ]

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        crop_h=630,
        crop_w=1120,
        patch_size=14,
        interpolation="bilinear",
    ):
        super(DefaultImagePointDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list, self.split_list = self.get_data_list()
        if isinstance(self.data_list, dict):
            self.data_name = list(self.data_list.keys())
        elif isinstance(self.data_list[0], dict):
            self.data_name = list([data["token"] for data in self.data_list])
        else:
            self.data_name = self.data_list
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_name), self.loop, os.path.basename(self.data_root), split
            )
        )

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.patch_size = patch_size
        self.patch_h = crop_h // patch_size
        self.patch_w = crop_w // patch_size
        self.transform_img = T.Compose(
            [
                T.Resize(
                    (self.patch_h * self.patch_size, self.patch_w * self.patch_size),
                    interpolation=INTERPOLATION_MODE[interpolation],
                ),
                T.ToTensor(),
            ]
        )

    def resize_correspondence_info(
        self, correspondence, size, size0, crop_size, _alignment
    ):
        h, w = size
        h0, w0 = size0
        left, top, right, bottom = crop_size
        crop_h = bottom - top
        crop_w = right - left
        mask_crop = (
            (correspondence[:, 1] >= top)
            & (correspondence[:, 1] < bottom)
            & (correspondence[:, 0] >= left)
            & (correspondence[:, 0] < right)
        )
        correspondence = correspondence[mask_crop]
        correspondence[:, 1] -= top
        correspondence[:, 0] -= left
        correspondence[:, 1] = (correspondence[:, 1] * h / crop_h // _alignment).astype(
            np.int32
        )
        correspondence[:, 0] = (correspondence[:, 0] * w / crop_w // _alignment).astype(
            np.int32
        )
        correspondence = correspondence[:, [1, 0, 2]]
        correspondence = np.unique(correspondence, axis=0)
        return correspondence

    def get_data_list(self):
        split_list = {}
        if isinstance(self.split, str):
            data_path = os.path.join(self.data_root, "splits", f"{self.split}.json")
            with open(data_path, "r", encoding="utf-8") as file:
                data_list = json.load(file)
            split_list[self.split] = list(data_list.keys())
        elif isinstance(self.split, Sequence):
            data_list = {}
            for split in self.split:
                data_path = os.path.join(self.data_root, "splits", f"{split}.json")
                with open(data_path, "r", encoding="utf-8") as file:
                    data_split_dict = json.load(file)
                    data_list.update(data_split_dict)
                split_list[split] = list(data_split_dict.keys())
        else:
            raise NotImplementedError
        return data_list, split_list

    def get_data_name(self, idx):
        return self.data_name[idx % len(self.data_name)]

    def get_split_name(self, idx):
        for split, names in self.split_list.items():
            if self.data_name[idx % len(self.data_name)] in names:
                return split
        return None

    def get_data(self, idx):
        data_dict = {}
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        data_dict["name"] = name
        data_dict["split"] = split
        data_path = self.data_list[name]

        pointclouds_path = data_path["pointclouds"]
        assets = os.listdir(pointclouds_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.PC_VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(pointclouds_path, asset))
        imgs_path = data_path["images"]
        imgs = [Image.open(asset) for asset in imgs_path]
        img_width, img_height = imgs[0].size
        div_w = img_width // self.patch_w
        div_h = img_height // self.patch_h
        div_min = max(min(div_w, div_h), 1)
        crop_img_width = div_min * self.patch_w
        crop_img_height = div_min * self.patch_h
        left = int((img_width - crop_img_width) / 2)
        top = int((img_height - crop_img_height) / 2)
        right = int((img_width + crop_img_width) / 2)
        bottom = int((img_height + crop_img_height) / 2)
        imgs = [img.crop((left, top, right, bottom)) for img in imgs]
        imgs = [self.transform_img(img) for img in imgs]
        if len(imgs) > 0:
            imgs_list = torch.stack(imgs)
            data_dict["images"] = imgs_list.float()
        else:
            data_dict["images"] = torch.empty(
                (0, 3, self.patch_h * self.patch_size, self.patch_w * self.patch_size)
            )
        data_dict["img_num"] = np.array([data_dict["images"].shape[0]], dtype=np.int32)

        correspondences_path = data_path["correspondences"]
        correspondence_infos = np.ones(
            (data_dict["coord"].shape[0], len(correspondences_path), 2), dtype=np.int32
        ) * (-1)
        for asset_id, asset in enumerate(correspondences_path):
            correspondence_info = np.load(asset).astype(np.int32)
            if np.array_equal(correspondence_info, -np.ones((1, 3))):
                continue
            correspondence_info = self.resize_correspondence_info(
                correspondence_info,
                (self.patch_h * self.patch_size, self.patch_w * self.patch_size),
                (img_height, img_width),
                (left, top, right, bottom),
                self.patch_size,
            )
            correspondence_infos[correspondence_info[:, -1], asset_id, :] = (
                correspondence_info[:, :-1]
            )
        data_dict["correspondence"] = correspondence_infos  # .reshape(-1, 2)

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

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ConcatDataset(Dataset):
    def __init__(self, datasets, loop=1):
        super(ConcatDataset, self).__init__()
        self.datasets = [build_dataset(dataset) for dataset in datasets]
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in the concat set.".format(
                len(self.data_list), self.loop
            )
        )

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i]), dtype=int) * i,
                    np.arange(len(self.datasets[i])),
                )
            )
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
