"""
AEO Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class AEODataset(DefaultDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        label_mapping = np.ones(41, dtype=int) * -1
        label_mapping[[0, 1, 3, 4, 13, 16, 19, 21, 22, 28, 29, 34, 36, 37, 38, 39]] = (
            np.arange(16)
        )
        self.label_mapping = label_mapping

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        data_dict["segment"] = self.label_mapping[data_dict["segment"]]
        return data_dict
