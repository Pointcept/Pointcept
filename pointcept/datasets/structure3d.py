"""
Structured3D Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
from collections.abc import Sequence

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Structured3DDataset(DefaultDataset):
    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*/*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*/*.pth"))
        else:
            raise NotImplementedError
        return data_list