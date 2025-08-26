"""
SNCF MLS Dataset

Author: Your Name
Adapted from: Pointcept NuScenesDataset
"""

import os
import numpy as np
import pickle
from collections.abc import Sequence

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SNCFMLSDataset(DefaultDataset):
    CLASSES = [
        "ground",
        "vegetation",
        "rail",
        "poles",
        "wires",
        "signalling",
        "fences",
        "installation",
    ]

    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        return os.path.join(self.data_root, "pointcept_dataset_tiles", f"infos_{split}.pkl")

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        tile_path = os.path.join(
            self.data_root, "pointcept_dataset_tiles", data["path"]
        )
        npz = np.load(tile_path)

        coord = npz["points"].astype(np.float32)
        segment = npz["labels"].astype(np.int64).reshape(-1)

        # Sanity check: must be in 0..num_classes-1 or ignore_index
        uniq = np.unique(segment)
        assert np.all((uniq == self.ignore_index) | ((uniq >= 0) & (uniq < len(self.CLASSES)))), \
            f"Invalid labels in {tile_path}: {uniq}"

        data_dict = dict(
            coord=coord,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        path = self.data_list[idx % len(self.data_list)]["path"]
        return os.path.basename(path)

    @staticmethod
    def get_learning_map(ignore_index):
        # Default pass-through; override when needed
        return {}
