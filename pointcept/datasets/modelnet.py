"""
ModelNet40 Dataset

get sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape)
at "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class ModelNetDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/modelnet40_normal_resampled',
                 class_names=None,
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 cache_data=False,
                 loop=1):
        super(ModelNetDataset, self).__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1  # force make loop = 1 while in test mode
        self.cache_data = cache_data
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}
        if test_mode:
            # TODO: Optimize
            pass

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info("Totally {} x {} samples in {} set.".format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(self.data_root, 'modelnet40_{}.txt'.format(self.split))
        data_list = np.loadtxt(split_path, dtype="str")
        return data_list

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        if self.cache_data:
            coord, normal, category = self.cache[data_idx]
        else:
            data_shape = '_'.join(self.data_list[data_idx].split('_')[0: -1])
            data_path = os.path.join(self.data_root, data_shape, self.data_list[data_idx] + '.txt')
            data = np.loadtxt(data_path, delimiter=',').astype(np.float32)
            coord, normal = data[:, 0:3], data[:, 3:6]
            category = np.array([self.class_names[data_shape]])
            if self.cache_data:
                self.cache[data_idx] = (coord, normal, category)
        data_dict = dict(coord=coord, normal=normal, category=category)
        return data_dict

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_list)
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
