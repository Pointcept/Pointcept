"""
Habitat-Matterport 3D Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os
from collections.abc import Sequence
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class HM3DDataset(DefaultDataset):
    def __init__(
        self,
        force_label=True,
        **kwargs,
    ):
        # if force_label, only load data with label
        self.force_label = force_label
        super().__init__(**kwargs)

    def get_single_data_list(self, split):
        if self.force_label:
            data_list = glob.glob(
                os.path.join(self.data_root, split, "*", "segment.npy")
            )
            data_list = [os.path.dirname(data) for data in data_list]
        else:
            data_list = glob.glob(os.path.join(self.data_root, split, "*"))
        return data_list

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = self.get_single_data_list(self.split)
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += self.get_single_data_list(split)
        else:
            raise NotImplementedError
        return data_list
