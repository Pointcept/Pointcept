"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/semantic_kitti',
                 learning_map=None,
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 loop=1):
        super(SemanticKITTIDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.learning_map = learning_map
        self.split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        )
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1    # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        if isinstance(self.split, str):
            seq_list = self.split2seq[split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += self.split2seq[split]
        else:
            raise NotImplementedError

        self.data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(
                os.listdir(os.path.join(seq_folder, "velodyne")))
            self.data_list += [os.path.join(seq_folder, "velodyne", file) for file in seq_files]
        logger = get_root_logger()
        logger.info("Totally {} x {} samples in {} set.".format(len(self.data_list), self.loop, split))

    def prepare_train_data(self, idx):
        name = self.get_data_name(idx)
        # load data
        data_idx = idx % len(self.data_list)
        with open(self.data_list[data_idx], 'rb') as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = self.data_list[data_idx].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        segment = np.vectorize(self.learning_map.__getitem__)(segment & 0xFFFF).astype(np.int64)
        data_dict = dict(coord=coord, strength=strength, segment=segment)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_idx = idx % len(self.data_list)
        with open(self.data_list[data_idx], 'rb') as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = self.data_list[data_idx].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            segment = np.zeros(coord.shape[0]).astype(np.int32)
        segment = np.vectorize(self.learning_map.__getitem__)(segment & 0xFFFF).astype(np.int64)

        data_dict = dict(coord=coord, strength=strength, segment=segment.astype(np.int64))

        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(
                aug(deepcopy(data_dict))
            )

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx))
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
