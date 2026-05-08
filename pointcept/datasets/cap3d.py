"""
Objaverse Point Cloud Dataset Loader

Load pre-processed .pt files.
Each .pt file is assumed to be a dictionary of PyTorch tensors.
"""

import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import json
from collections.abc import Sequence
import random

from .builder import DATASETS
from .defaults import DefaultDataset, DefaultImagePointDataset


@DATASETS.register_module()
class Cap3DDataset(DefaultDataset):
    """
    Cap3D Dataset for loading .pt files.
    """

    def __init__(
        self,
        data_num=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if data_num:
            self.data_list = self.data_list[:data_num]

    @staticmethod
    def get_normals(center, coords):
        Cs = np.repeat(center.reshape((1, -1)), coords.shape[0], axis=0)
        view_dirs = coords - Cs
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        dot_product = np.sum(normals * view_dirs, axis=-1)
        flip_mask = dot_product < 0
        normals[flip_mask] = -normals[flip_mask]

        # Normalize normals a nd m
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    def get_data(self, idx):
        """
        The core data loading function.

        Args:
            idx (int): The index of the data sample.

        Returns:
            dict: A dictionary containing the loaded and processed point cloud data.
        """
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        try:
            data = torch.load(data_path, map_location="cpu")
        except Exception as e:
            raise IOError(f"Failed to load a .pt file at {data_path}. Error: {e}")

        data = data.numpy().transpose()
        color = data[:, 3:]
        coord = data[:, :3]
        normal = self.get_normals(np.mean(coord, axis=0), coord)

        data_dict = {}

        data_dict["coord"] = coord
        data_dict["color"] = color
        data_dict["normal"] = normal

        data_dict["name"] = name
        data_dict["split"] = split

        return data_dict


@DATASETS.register_module()
class Cap3DImagePointDataset(DefaultImagePointDataset):
    def __init__(
        self,
        data_num=-1,
        **kwargs,
    ):
        self.data_num = data_num
        super().__init__(**kwargs)

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

        if self.data_num < 0 or self.data_num >= len(data_list):
            selected_data_list = data_list
            selected_split_list = split_list
            return selected_data_list, selected_split_list

        all_keys = list(data_list.keys())
        selected_keys = random.sample(all_keys, self.data_num)
        selected_data_list = {key: data_list[key] for key in selected_keys}
        valid_keys = set(selected_keys)

        selected_split_list = {}
        for split_name, key_list in split_list.items():
            filtered_keys = [key for key in key_list if key in valid_keys]
            if filtered_keys:
                selected_split_list[split_name] = filtered_keys
        return selected_data_list, selected_split_list

    @staticmethod
    def get_normals(center, coords):
        Cs = np.repeat(center.reshape((1, -1)), coords.shape[0], axis=0)
        view_dirs = coords - Cs
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        dot_product = np.sum(normals * view_dirs, axis=-1)
        flip_mask = dot_product < 0
        normals[flip_mask] = -normals[flip_mask]
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    def get_pc_data(self, data_path):
        try:
            data = torch.load(data_path, map_location="cpu")
        except Exception as e:
            raise IOError(f"Failed to load a .pt file at {data_path}. Error: {e}")

        data = data.numpy().transpose()
        color = data[:, 3:]
        coord = data[:, :3]
        normal = self.get_normals(np.mean(coord, axis=0), coord)

        data_dict = {}

        data_dict["coord"] = coord
        data_dict["color"] = color
        data_dict["normal"] = normal

        return data_dict

    def get_data(self, idx):
        data_dict = {}
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        data_dict["name"] = name
        data_dict["split"] = split
        data_path = self.data_list[name]

        pointclouds_path = data_path["pointclouds"]
        pc_dict = self.get_pc_data(pointclouds_path)
        data_dict.update(pc_dict)
        imgs_path = data_path["images"]
        imgs = [Image.open(asset).convert("RGB") for asset in imgs_path]
        if len(imgs) > 0:
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
            imgs_list = torch.stack(imgs)
            data_dict["images"] = imgs_list.float()
        else:
            data_dict["images"] = torch.empty(
                (0, 3, self.patch_h * self.patch_size, self.patch_w * self.patch_size)
            )
        data_dict["img_num"] = np.array([data_dict["images"].shape[0]], dtype=np.int32)

        correspondences_path = data_path["correspondences"]
        correspondence_infos = np.ones(
            (data_dict["coord"].shape[0], len(correspondences_path), 2),
            dtype=np.float32,
        ) * (-1)
        for asset_id, asset in enumerate(correspondences_path):
            correspondence_info = np.load(asset).astype(np.float32)
            if np.array_equal(correspondence_info, -np.ones((1, 3))):
                continue
            correspondence_info = self.resize_correspondence_info(
                correspondence_info,
                (self.patch_h * self.patch_size, self.patch_w * self.patch_size),
                (img_height, img_width),
                (left, top, right, bottom),
                self.patch_size,
            )
            correspondence_infos[
                correspondence_info[:, -1].astype(np.int32), asset_id, :
            ] = correspondence_info[:, :-1]
        data_dict["correspondence"] = correspondence_infos

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
