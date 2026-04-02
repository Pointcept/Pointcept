"""
ScanObjectNN Dataset

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import h5py
import numpy as np
import torch
from collections.abc import Sequence
import open3d as o3d
import copy

from pointcept.utils.logger import get_root_logger

from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose


@DATASETS.register_module()
class ScanObjectNNRawDataset(DefaultDataset):
    """
    ScanObjectNN Dataset loaded from a standard folder structure.
    """

    def __init__(self, if_img=True, crop_h=518, crop_w=518, patch_size=14, **kwargs):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.patch_size = patch_size
        self.patch_h = crop_h // patch_size
        self.patch_w = crop_w // patch_size
        self.if_img = if_img
        super().__init__(**kwargs)

    def get_data_list(self):
        """
        Overrides the parent method.
        Scans the data directory (or directories) to find all relevant .bin files
        for the specified split(s).
        """
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        logger = get_root_logger()

        for split in split_list:
            split_path = os.path.join(self.data_root, split)
            if not os.path.isdir(split_path):
                raise FileNotFoundError(f"Split directory not found at: {split_path}")

            logger.info(f"Scanning for .bin files in {split_path}")
            # Use glob to find all .bin files within the class subdirectories
            all_bin_files = glob.glob(os.path.join(split_path, "*", "*.bin"))

            # Filter out auxiliary files, keeping only the main object files
            filtered_files = [
                path
                for path in all_bin_files
                if not path.endswith("_part.bin") and not path.endswith("_indices.bin")
            ]

            if not filtered_files:
                logger.warning(
                    f"No valid .bin files found in {split_path}. Please check the file structure."
                )

            data_list.extend(filtered_files)

        return sorted(data_list)  # Sort for reproducibility

    def get_data(self, idx):
        """
        Overrides the parent method.
        Loads and parses a single .bin file based on its path.
        The logic for parsing the binary file format remains the same.
        """
        data_path = self.data_list[idx % len(self.data_list)]

        try:
            raw_data = np.fromfile(data_path, dtype=np.float32)
        except IOError as e:
            logger = get_root_logger()
            logger.error(f"Could not read file {data_path}: {e}")
            # Skip to the next sample if the current one is unreadable
            return self.get_data((idx + 1) % len(self.data_list))

        if raw_data.size == 0:
            logger = get_root_logger()
            logger.warning(f"File is empty, skipping: {data_path}")
            return self.get_data((idx + 1) % len(self.data_list))

        # First float is the number of points
        num_points = int(raw_data[0])
        point_data_flat = raw_data[1:]

        # Validate that the file is not corrupt
        expected_size = num_points * 11
        if point_data_flat.size != expected_size:
            logger = get_root_logger()
            logger.warning(
                f"Data corruption detected in {data_path}: "
                f"Expected {expected_size} floats for {num_points} points, but found {point_data_flat.size}. Skipping."
            )
            return self.get_data((idx + 1) % len(self.data_list))

        # Reshape into a (num_points, 11) array
        point_cloud = point_data_flat.reshape(num_points, 11)
        rot_mat = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        rotated_coord = point_cloud[:, :3].astype(np.float32) @ rot_mat.T
        rotated_normal = point_cloud[:, 3:6].astype(np.float32) @ rot_mat.T
        if self.if_img:
            imgs_tensor = torch.empty(
                (0, 3, self.patch_h * self.patch_size, self.patch_w * self.patch_size)
            )
            img_num = np.array([imgs_tensor.shape[0]], dtype=np.int32)
            correspondence_infos = np.ones(
                (point_cloud.shape[0], imgs_tensor.shape[0], 2),
                dtype=np.float32,
                # (data_dict["coord"].shape[0], len(correspondences_path), 2), dtype=np.int32
            ) * (-1)
            # Pack the data into the dictionary format expected by the framework
            # Attributes order: x, y, z, nx, ny, nz, r, g, b, instance_label, semantic_label
            data_dict = {
                "coord": rotated_coord.astype(np.float32),
                "normal": rotated_normal.astype(np.float32),
                "color": point_cloud[:, 6:9].astype(np.float32),
                "instance": point_cloud[:, 9].astype(np.int64),
                "segment": point_cloud[:, 10].astype(np.int64),
                "name": self.get_data_name(idx),
                "split": self.get_split_name(idx),
                "images": imgs_tensor,
                "img_num": img_num,
                "correspondence": correspondence_infos,
            }
        else:
            data_dict = {
                "coord": rotated_coord.astype(np.float32),
                "normal": rotated_normal.astype(np.float32),
                "color": point_cloud[:, 6:9].astype(np.float32),
                "instance": point_cloud[:, 9].astype(np.int64),
                "segment": point_cloud[:, 10].astype(np.int64),
                "name": self.get_data_name(idx),
                "split": self.get_split_name(idx),
            }
        return data_dict

    def get_data_name(self, idx):
        """
        Overrides the parent method.
        Returns the filename without extension as the data name.
        Example: '005_00001'
        """
        path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(path))[0]

    def get_split_name(self, idx):
        """
        Overrides the parent method.
        Correctly extracts the split name (e.g., 'train') from the file path.
        """
        path = self.data_list[idx % len(self.data_list)]
        return os.path.basename(os.path.dirname(os.path.dirname(path)))


@DATASETS.register_module()
class ScanObjectNNDataset(DefaultDataset):
    """
    ScanObjectNN Dataset.
    Loads data from HDF5 files into memory.
    """

    def __init__(
        self,
        class_names,
        if_color=True,
        if_normal=True,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
    ):
        self.if_color = if_color
        self.if_normal = if_normal
        self.class_names = class_names
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
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop, os.path.basename(self.data_root), split
            )
        )

    @staticmethod
    def get_normals(center, coords):
        Cs = np.repeat(center.reshape((1, -1)), coords.shape[0], axis=0)
        view_dirs = coords - Cs
        view_dirs = view_dirs / (
            np.linalg.norm(view_dirs, axis=-1, keepdims=True) + 1e-6
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        dot_product = np.sum(normals * view_dirs, axis=-1)
        flip_mask = dot_product < 0
        normals[flip_mask] = -normals[flip_mask]
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    def get_data_list(self):
        """
        Called by the parent constructor.
        Here, we load the HDF5 file into memory and return a list of sample indices.
        """
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        self.points = []
        self.labels = []
        for split in split_list:
            if split == "train":
                h5_path = os.path.join(self.data_root, "training_objectdataset.h5")
            elif split == "test":
                h5_path = os.path.join(self.data_root, "test_objectdataset.h5")
            else:
                raise NotImplementedError(f"Split {self.split} not supported.")

            with h5py.File(h5_path, "r") as h5:
                self.points.append(np.array(h5["data"]).astype(np.float32))
                # Use np.int64 for labels, as it's the default for PyTorch's CrossEntropyLoss
                self.labels.append(np.array(h5["label"]).astype(np.int64))
        self.points = np.concatenate(self.points, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        # The data_list is now a list of indices, from 0 to num_samples-1.
        return list(range(self.points.shape[0]))

    def get_data(self, idx):
        """
        Overrides the parent method.
        Instead of loading from a file path, we get the data from the pre-loaded numpy arrays.
        """
        data_idx = self.data_list[idx % len(self.data_list)]
        coord = self.points[data_idx].copy()
        if self.if_color:
            color = np.zeros_like(coord)
        if self.if_normal:
            normal = self.get_normals(np.mean(coord, axis=0), coord)
        label = self.labels[data_idx]
        category = np.array([label])

        data_dict = {
            "coord": coord,
            "color": color,
            "normal": normal,
            "category": category,
            "name": self.get_data_name(idx),
        }

        return data_dict

    def get_data_name(self, idx):
        """
        Overrides the parent method.
        Returns a unique name for the sample at the given index.
        """
        real_idx = idx % len(self.data_list)
        return f"{self.split}-{real_idx}"

    def get_split_name(self, idx):
        return self.split

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        category = data_dict.pop("category")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(copy.deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])
        data_dict = dict(
            voting_list=data_dict_list,
            category=category,
            name=self.get_data_name(idx),
        )
        return data_dict


@DATASETS.register_module()
class ScanObjectNNHardestDataset(ScanObjectNNDataset):
    """
    ScanObjectNN "Hardest" variant.
    Inherits from our refactored ScanObjectNN and only overrides the file loading logic.
    """

    def get_data_list(self):
        """
        Overrides the parent method to load from the augmented HDF5 files.
        """
        if self.split == "train":
            h5_path = os.path.join(
                self.data_root, "training_objectdataset_augmentedrot_scale75.h5"
            )
        elif self.split == "test":
            h5_path = os.path.join(
                self.data_root, "test_objectdataset_augmentedrot_scale75.h5"
            )
        else:
            raise NotImplementedError(f"Split {self.split} not supported.")

        with h5py.File(h5_path, "r") as h5:
            self.points = np.array(h5["data"]).astype(np.float32)
            self.labels = np.array(h5["label"]).astype(np.int64)

        return list(range(self.points.shape[0]))
