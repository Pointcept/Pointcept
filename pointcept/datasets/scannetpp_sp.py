"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import os
import numpy as np
import glob

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from pointcept.datasets.scannetpp import ScanNetPPDataset

from configs._base_.dataset.scannetpp import CLASS_LABELS_PP, INST_LABELS_PP


@DATASETS.register_module()
class ScanNetPPSpDataset(ScanNetPPDataset):

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "instance",
        "superpoint",
    ]
    class2id = np.array([CLASS_LABELS_PP.index(c) for c in INST_LABELS_PP])

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
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

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "superpoint" in data_dict.keys():
            data_dict["superpoint"] = data_dict["superpoint"].astype(np.int32)

        if not self.multilabel:
            if "segment" in data_dict.keys():
                if "vtx" in self.split:
                    data_dict["segment"] = (
                        data_dict["segment"].reshape([-1]).astype(np.int32)
                    )
                else:
                    data_dict["segment"] = data_dict["segment"][:, 0].astype(np.int32)
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                if "vtx" in self.split:
                    data_dict["instance"] = (
                        data_dict["instance"].reshape([-1]).astype(np.int32)
                    )
                else:
                    data_dict["instance"] = data_dict["instance"][:, 0].astype(np.int32)
            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        data_dict = self.transform(data_dict)
        data_dict = self.test_voxelize(data_dict)
        if self.test_crop:
            data_dict = self.test_crop(data_dict)
        data_dict = self.post_transform(data_dict)
        data_dict = dict(
            data_dict=data_dict,
            segment=segment,
            instance=instance,
            name=self.get_data_name(idx),
        )
        return data_dict

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        name = data_dict["name"]
        data_dict = self.transform(data_dict)
        data_dict["name"] = name
        return data_dict
