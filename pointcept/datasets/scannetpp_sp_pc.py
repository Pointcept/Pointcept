"""
ScanNet++ dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from .defaults import DefaultDataset
from .builder import DATASETS
import random
import os
import numpy as np
import glob
import glob
import math
import numpy as np
import os.path as osp
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union
import os
from pointcept.models.spformer.utils import Instances3D
from torch import Tensor

from pointcept.datasets.transform import ToTensor
from configs._base_.dataset.scannetpp import CLASS_LABELS_PP, INST_LABELS_PP
from pointcept.utils.cache import shared_dict

try:
    import pointgroup_ops_sp.sp_ops as pointops
except ImportError:
    pointops = None


@DATASETS.register_module()
class ScanNetPPSPPCDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "instance",
        "superpoint",
    ]

    def __init__(
        self,
        multilabel=False,
        voxel_cfg=None,
        training=True,
        with_label=True,
        mode=4,
        with_elastic=True,
        use_xyz=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            pointops is not None
        ), "Please install the pointgoup_ops_sp from the lib folder"
        from pointcept.models.sgiFormer.transform_tensor import (
            MeanShiftT,
            RandomFlipT,
            RandomRotateT,
            RandomDropoutT,
            RandomScaleT,
            RandomTranslationT,
            CustomElasticDistortionT,
            ChromaticAutoContrastT,
            ChromaticTranslationT,
            ChromaticJitterT,
            SphereCropT,
        )

        self.multilabel = multilabel
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.use_xyz = use_xyz
        # self.data_list = self.filenames
        self.ins_cls_ids = [CLASS_LABELS_PP.index(c) for c in INST_LABELS_PP]
        self.inclassT = InsClassMapT(ins_cls_ids=self.ins_cls_ids)
        transform = [
            dict(type=ToTensor),
            dict(type=MeanShiftT),
            dict(type=RandomDropoutT, dropout_ratio=0.2, dropout_application_ratio=0.5),
            dict(type=RandomFlipT, p=0.5),
            dict(type=RandomRotateT, angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.95),
            dict(type=RandomRotateT, angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type=RandomRotateT, angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type=RandomScaleT, scale=[0.8, 1.2]),
            dict(type=RandomTranslationT, shift=[0.1, 0.1, 0.1]),
            dict(
                type=CustomElasticDistortionT,
                distortion_params=[[10, 60], [30, 180]],
                p=0.5,
            ),
            dict(type=ChromaticAutoContrastT, p=0.2, blend_factor=None),
            dict(type=ChromaticTranslationT, p=0.95, ratio=0.1),
            dict(type=ChromaticJitterT, p=0.95, std=0.05),
            dict(type=SphereCropT, sample_rate=0.8, mode="random"),
        ]

        self.train_aug = []

        for item in transform:
            cls = item["type"]  # Extract class
            # Extract arguments
            kwargs = {k: v for k, v in item.items() if k != "type"}
            obj = cls(**kwargs)  # Initialize object
            self.train_aug.append(obj)

    def load(self, idx):
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
                data_dict["segment"] = data_dict["segment"][:, 0].astype(np.int32)
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][:, 0].astype(np.int32)

            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError

        return (
            data_dict["coord"],
            data_dict["color"],
            data_dict["superpoint"],
            data_dict["segment"],
            data_dict["instance"],
        )

    def get_data(self, idx):
        scan_id = self.get_data_name(idx)
        data = self.load(idx)
        data = (
            self.transform_train(*data) if self.training else self.transform_test(*data)
        )
        # data = self.transform_test(*data)
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = data

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        semantic_label = torch.from_numpy(semantic_label).long()
        # semantic_label = torch.where(
        #     semantic_label < 2, -100, semantic_label - 2)
        semantic_label = self.inclassT.get_mapped_segment(semantic_label)
        instance_label = torch.from_numpy(instance_label).long()
        inst = self.get_instance3D(instance_label, semantic_label, superpoint, scan_id)
        return (
            scan_id,
            coord,
            coord_float,
            feat,
            superpoint,
            inst,
            semantic_label,
            instance_label,
        )

    def data_aug(self, xyz, rgb, superpoint, semantic_label, instance_label):
        data_dict = {
            "coord": xyz,
            "color": rgb,
            "superpoint": superpoint,
            "segment": semantic_label,
            "instance": instance_label,
        }
        for transform in self.train_aug:
            data_dict = transform(data_dict)
        return (
            data_dict["coord"].numpy().astype(np.float32),
            data_dict["color"].numpy().astype(np.float32),
            data_dict["superpoint"].numpy().astype(np.int32),
            data_dict["segment"].numpy().astype(np.int32),
            data_dict["instance"].numpy().astype(np.int32),
        )

    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle, rgb, superpoint, semantic_label, instance_label = self.data_aug(
            xyz, rgb, superpoint, semantic_label, instance_label
        )
        rgb = rgb / 127.5 - 1
        xyz = xyz_middle * self.voxel_cfg["scale"]
        xyz = xyz - xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz -= xyz.mean(0)
        xyz_middle = xyz
        rgb = rgb / 127.5 - 1
        xyz = xyz_middle * self.voxel_cfg["scale"]
        xyz = xyz - xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        (
            scan_ids,
            coords,
            coords_float,
            feats,
            superpoints,
            insts,
            segments,
            instance_labels,
        ) = ([], [], [], [], [], [], [], [])
        batch_offsets = [0]
        superpoint_bias = 0

        for i, data in enumerate(batch):
            (
                scan_id,
                coord,
                coord_float,
                feat,
                superpoint,
                inst,
                segment,
                instance_label,
            ) = data

            superpoint += superpoint_bias
            superpoint_bias = superpoint.max().item() + 1 if len(superpoint) > 0 else 1
            batch_offsets.append(superpoint_bias)
            # print(superpoint_bias)

            scan_ids.append(scan_id)
            coords.append(
                torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1)
            )
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            insts.append(inst)
            segments.append(segment)
            instance_labels.append(instance_label)

        # merge all scan in batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords = torch.cat(coords, 0)
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        segments = torch.cat(segments, 0)
        instance_labels = torch.cat(instance_labels, 0)
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)

        # voxelize
        spatial_shape = np.clip(
            (coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg["spatial_shape"][0], None
        )  # long [3]

        voxel_coords, p2v_map, v2p_map = pointops.voxelization_idx(
            coords, len(batch), self.mode
        )

        return {
            "scan_ids": scan_ids,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "spatial_shape": spatial_shape,
            "feats": feats,
            "superpoints": superpoints,
            "batch_offsets": batch_offsets,
            "insts": insts,
            "name": scan_ids,
            "origin_segment": segments,
            "origin_instance": instance_labels,
        }

    def get_cropped_inst_label(
        self, instance_label: np.ndarray, valid_idxs: np.ndarray
    ) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]

        # Check for empty labels
        if instance_label.size == 0:
            print("Warning: Empty instance_label encountered!")
            return instance_label  # Return as-is, or handle appropriately

            # Compact the labels
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1

        return instance_label

    def get_instance3D(self, instance_label, semantic_label, superpoint, scan_id):
        if instance_label.numel() == 0:
            dummy_inst = Instances3D(0, get_instances=np.array([]))
            dummy_inst.gt_labels = torch.Tensor([]).long()
            dummy_inst.gt_spmasks = torch.Tensor([])
            return dummy_inst
        num_insts = instance_label.max().item() + 1
        num_points = len(instance_label)
        gt_masks, gt_labels = [], []

        gt_inst = torch.zeros(num_points, dtype=torch.int64)
        for i in range(num_insts):
            idx = torch.where(instance_label == i)
            assert len(torch.unique(semantic_label[idx])) == 1
            sem_id = semantic_label[idx][0]
            if semantic_label[idx][0] == -1:
                # sem_id = 1
                # gt_inst[idx] = sem_id * 1000 + i + 1
                continue
            gt_mask = torch.zeros(num_points)
            gt_mask[idx] = 1
            gt_masks.append(gt_mask)
            gt_label = sem_id
            gt_labels.append(gt_label)
            gt_inst[idx] = (sem_id + 1) * 1000 + i + 1
        if gt_masks:
            gt_masks = torch.stack(gt_masks, dim=0)
            gt_spmasks = torch_scatter.scatter_mean(
                gt_masks.float(), superpoint, dim=-1
            )
            gt_spmasks = (gt_spmasks > 0.5).float()
        else:
            gt_spmasks = torch.tensor([])
        gt_labels = torch.tensor(gt_labels)

        inst = Instances3D(num_points, gt_instances=gt_inst.numpy())
        inst.gt_labels = gt_labels.long()
        inst.gt_spmasks = gt_spmasks
        return inst

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        return data_dict


@DATASETS.register_module()
class ScanNetPPSPPCDatasetTrainSample(ScanNetPPSPPCDataset):
    def __init__(
        self,
        nsamples=30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nsamples = nsamples

    def __len__(self):
        return self.nsamples


class InsClassMapT(object):

    def __init__(self, ins_cls_ids):
        self.ins_cls_ids = ins_cls_ids
        self.id_to_index = {cls_id: i for i, cls_id in enumerate(self.ins_cls_ids)}
        self.index_to_id = {i: cls_id for i, cls_id in enumerate(self.ins_cls_ids)}

    def get_mapped_segment(self, seg):
        new_segment = torch.ones_like(seg) * -1
        for i, ins_cls_id in enumerate(self.ins_cls_ids):
            new_segment[seg == ins_cls_id] = i
        seg = new_segment
        return seg

    def reverse_map(self, predicted_classes: Tensor) -> Tensor:
        vectorized_map = np.vectorize(lambda x: self.index_to_id.get(x, -1))
        return vectorized_map(predicted_classes)
